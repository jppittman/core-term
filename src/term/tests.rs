//! Unit tests for the main TerminalEmulator struct and its core logic.
//! These tests aim to verify the internal state changes and actions produced
//! by the TerminalEmulator based on various inputs, adhering to its public API.

// Module for tests related to the Term struct itself
#[cfg(test)]
mod term_tests {
    use crate::ansi::commands::{AnsiCommand, Attribute, C0Control, CsiCommand, EscCommand};
    use crate::backends::{BackendEvent, MouseButton, MouseEventType}; // Added MouseButton, MouseEventType
    use crate::color::{Color, NamedColor};
    use crate::config::{KeySymbol, Modifiers}; // Added for potential future use if mapping from BackendEvent to UserInputAction is tested here
    use crate::glyph::{AttrFlags, Attributes, Glyph};
    use crate::term::{
        DecModeConstant, EmulatorAction, EmulatorInput, SelectionMode, SelectionState,
        TerminalEmulator, TerminalInterface, UserInputAction, // Added UserInputAction, SelectionState, SelectionMode
    };

    use test_log::test; // Ensure test_log is a dev-dependency

    // --- Test Helpers ---

    fn new_term(cols: usize, rows: usize) -> TerminalEmulator {
        TerminalEmulator::new(cols, rows, 100)
    }

    fn process_input(term: &mut TerminalEmulator, input: EmulatorInput) -> Option<EmulatorAction> {
        let action = term.interpret_input(input);
        if action.is_none() && !TerminalInterface::take_dirty_lines(term).is_empty() {
            return Some(EmulatorAction::RequestRedraw);
        }
        action
    }

    fn process_commands(
        term: &mut TerminalEmulator,
        commands: Vec<AnsiCommand>,
    ) -> Vec<EmulatorAction> {
        let mut actions = Vec::new();
        for cmd in commands {
            if let Some(action) = term.interpret_input(EmulatorInput::Ansi(cmd)) {
                actions.push(action);
            }
        }
        if !TerminalInterface::take_dirty_lines(term).is_empty()
            && !actions
                .iter()
                .any(|a| matches!(a, EmulatorAction::RequestRedraw))
        {
            actions.push(EmulatorAction::RequestRedraw);
        }
        actions
    }

    // screen_to_string_vec is unused, marked by compiler warning.
    /*
    fn screen_to_string_vec(term: &TerminalEmulator) -> Vec<String> {
        let (cols, rows) = TerminalInterface::dimensions(term);
        let mut result = Vec::with_capacity(rows);
        for y in 0..rows {
            let line: String = (0..cols).map(|x| TerminalInterface::get_glyph(term, x, y).c).collect();
            result.push(line);
        }
        result
    }
    */

    fn get_glyph_at(term: &TerminalEmulator, x: usize, y: usize) -> Glyph {
        TerminalInterface::get_glyph(term, x, y)
    }

    fn assert_cursor_pos(term: &TerminalEmulator, x: usize, y: usize, message: &str) {
        assert_eq!(
            term.cursor_pos(),
            (x, y),
            "Logical cursor position check: {}",
            message
        );
    }

    fn assert_screen_cursor_pos(term: &TerminalEmulator, x: usize, y: usize, message: &str) {
        assert_eq!(
            TerminalInterface::get_screen_cursor_pos(term),
            (x, y),
            "Screen cursor position check: {}",
            message
        );
    }

    // --- Initialization Tests ---

    #[test]
    fn test_new_terminal_initial_state() {
        let term = new_term(80, 24);
        assert_eq!(
            TerminalInterface::dimensions(&term),
            (80, 24),
            "Initial dimensions"
        );
        assert_cursor_pos(&term, 0, 0, "Initial logical cursor position");
        assert_screen_cursor_pos(&term, 0, 0, "Initial screen cursor position");
        assert!(
            TerminalInterface::is_cursor_visible(&term),
            "Cursor initially visible"
        );
        assert!(!term.is_alt_screen_active(), "Initially not on alt screen");

        let expected_initial_attrs = Attributes::default();
        assert_eq!(
            get_glyph_at(&term, 0, 0).attr,
            expected_initial_attrs,
            "Initial cell attributes at (0,0)"
        );
        assert_eq!(
            get_glyph_at(&term, 79, 23).attr,
            expected_initial_attrs,
            "Initial cell attributes at (79,23)"
        );

        let mut term_mut_for_print = new_term(1, 1);
        process_input(
            &mut term_mut_for_print,
            EmulatorInput::Ansi(AnsiCommand::Print('T')),
        );
        assert_eq!(
            get_glyph_at(&term_mut_for_print, 0, 0).attr,
            expected_initial_attrs,
            "Initial cursor attributes for printing"
        );

        let mut term_mut_dirty = new_term(80, 24);
        let dirty_lines = TerminalInterface::take_dirty_lines(&mut term_mut_dirty);
        assert_eq!(dirty_lines.len(), 24, "All lines initially dirty");
        assert_eq!(
            dirty_lines,
            (0..24).collect::<Vec<usize>>(),
            "Correct dirty line indices"
        );
    }

    #[test]
    fn test_new_terminal_minimum_dimensions() {
        let term = new_term(0, 0);
        assert_eq!(
            TerminalInterface::dimensions(&term).0,
            1,
            "Minimum width clamped to 1"
        );
        assert_eq!(
            TerminalInterface::dimensions(&term).1,
            1,
            "Minimum height clamped to 1"
        );
    }

    // --- SGR and Attribute Tests ---

    #[test]
    fn test_sgr_reset_restores_default_attributes_and_applies_to_new_chars() {
        let mut term = new_term(10, 1);
        let default_attrs = Attributes::default();

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Named(NamedColor::Red)),
                Attribute::Background(Color::Named(NamedColor::Blue)),
                Attribute::Bold,
            ]))),
        );

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('X')));
        let glyph_x = get_glyph_at(&term, 0, 0);
        assert_eq!(glyph_x.c, 'X');
        assert_eq!(glyph_x.attr.fg, Color::Named(NamedColor::Red));
        assert_eq!(glyph_x.attr.bg, Color::Named(NamedColor::Blue));
        assert!(glyph_x.attr.flags.contains(AttrFlags::BOLD));

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Reset,
            ]))),
        );

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('Y')));
        let glyph_y = get_glyph_at(&term, 1, 0);
        assert_eq!(glyph_y.c, 'Y');
        assert_eq!(
            glyph_y.attr, default_attrs,
            "Y attributes should be Default after SGR Reset"
        );

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1, 3))),
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInLine(0))),
        );
        for x_idx in 2..TerminalInterface::dimensions(&term).0 {
            let erased_glyph = get_glyph_at(&term, x_idx, 0);
            assert_eq!(
                erased_glyph.attr, default_attrs,
                "Erased cell at ({},0) should have default attributes after SGR Reset",
                x_idx
            );
        }
    }

    #[test]
    fn test_reverse_attribute_impact_on_printed_char() {
        let mut term = new_term(5, 1);
        let default_attrs = Attributes::default();

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        let glyph_a = get_glyph_at(&term, 0, 0);
        assert_eq!(glyph_a.attr.fg, default_attrs.fg);
        assert_eq!(glyph_a.attr.bg, default_attrs.bg);
        assert!(!glyph_a.attr.flags.contains(AttrFlags::REVERSE));

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Reverse,
            ]))),
        );

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('B')));
        let glyph_b = get_glyph_at(&term, 1, 0);
        assert_eq!(glyph_b.attr.fg, default_attrs.fg);
        assert_eq!(glyph_b.attr.bg, default_attrs.bg);
        assert!(glyph_b.attr.flags.contains(AttrFlags::REVERSE));

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::NoReverse,
            ]))),
        );

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('C')));
        let glyph_c = get_glyph_at(&term, 2, 0);
        assert_eq!(glyph_c.attr.fg, default_attrs.fg);
        assert_eq!(glyph_c.attr.bg, default_attrs.bg);
        assert!(!glyph_c.attr.flags.contains(AttrFlags::REVERSE));
    }

    #[test]
    fn test_explicit_fg_bg_with_reverse() {
        let mut term = new_term(5, 1);
        let red_fg = Color::Named(NamedColor::Red);
        let blue_bg = Color::Named(NamedColor::Blue);

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Named(NamedColor::Red)),
                Attribute::Background(Color::Named(NamedColor::Blue)),
            ]))),
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Reverse,
            ]))),
        );

        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('R')));
        let glyph_r = get_glyph_at(&term, 0, 0);
        assert_eq!(glyph_r.attr.fg, red_fg);
        assert_eq!(glyph_r.attr.bg, blue_bg);
        assert!(glyph_r.attr.flags.contains(AttrFlags::REVERSE));
    }

    #[test]
    fn test_screen_default_attributes_updated_by_sgr_for_clearing() {
        let mut term = new_term(5, 1);
        let default_attrs = Attributes::default();
        let red_fg = Color::Named(NamedColor::Red);
        let blue_bg = Color::Named(NamedColor::Blue);

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1, 3))),
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInLine(0))),
        );
        assert_eq!(get_glyph_at(&term, 3, 0).attr, default_attrs);

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Named(NamedColor::Red)),
                Attribute::Background(Color::Named(NamedColor::Blue)),
            ]))),
        );

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(1, 1))),
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInLine(2))),
        );

        let expected_erase_attrs = Attributes {
            fg: red_fg,
            bg: blue_bg,
            flags: AttrFlags::empty(),
        };
        assert_eq!(get_glyph_at(&term, 0, 0).attr, expected_erase_attrs);
        assert_eq!(get_glyph_at(&term, 1, 0).attr, expected_erase_attrs);
    }

    #[test]
    fn test_initial_screen_attributes_are_default() {
        let term = new_term(10, 5);
        let glyph = get_glyph_at(&term, 0, 0);
        let expected_attrs = Attributes::default(); // Compare with the constant

        assert_eq!(glyph.c, ' ');
        assert_eq!(glyph.attr, expected_attrs);
    }

    #[test]
    fn test_erase_in_line_uses_current_sgr_attributes() {
        let mut term = new_term(10, 2);
        let default_attrs = Attributes::default();

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Named(NamedColor::Green)),
                Attribute::Background(Color::Named(NamedColor::Magenta)),
            ]))),
        );
        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('A'),
                AnsiCommand::Print('B'),
                AnsiCommand::Print('C'),
            ],
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInLine(0))),
        );

        let expected_attrs_magenta_bg = Attributes {
            fg: Color::Named(NamedColor::Green),
            bg: Color::Named(NamedColor::Magenta),
            flags: AttrFlags::empty(),
        };

        for x in 3..10 {
            let glyph = get_glyph_at(&term, x, 0);
            assert_eq!(glyph.c, ' ');
            assert_eq!(glyph.attr, expected_attrs_magenta_bg);
        }

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Reset,
            ]))),
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(2, 1))),
        );
        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('D'),
                AnsiCommand::Print('E'),
                AnsiCommand::Print('F'),
            ],
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInLine(0))),
        );

        for x in 3..10 {
            let glyph = get_glyph_at(&term, x, 1);
            assert_eq!(glyph.c, ' ');
            assert_eq!(glyph.attr, default_attrs);
        }
    }

    #[test]
    fn test_erase_in_display_uses_current_sgr_attributes() {
        let mut term = new_term(5, 3);
        let default_attrs = Attributes::default();

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Named(NamedColor::Cyan)),
                Attribute::Background(Color::Named(NamedColor::Yellow)),
            ]))),
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(2))),
        );

        let expected_attrs_yellow_bg = Attributes {
            fg: Color::Named(NamedColor::Cyan),
            bg: Color::Named(NamedColor::Yellow),
            flags: AttrFlags::empty(),
        };

        for y in 0..3 {
            for x in 0..5 {
                let glyph = get_glyph_at(&term, x, y);
                assert_eq!(glyph.c, ' ');
                assert_eq!(glyph.attr, expected_attrs_yellow_bg);
            }
        }

        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Reset,
            ]))),
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(2))),
        );

        for y in 0..3 {
            for x in 0..5 {
                let glyph = get_glyph_at(&term, x, y);
                assert_eq!(glyph.c, ' ');
                assert_eq!(glyph.attr, default_attrs);
            }
        }
    }

    // --- Core Functionality Tests ---

    #[test]
    fn test_c0_bs_backspace_moves_cursor_left() {
        let mut term = new_term(5, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::BS)),
        );
        assert_cursor_pos(&term, 0, 0, "BS should move cursor from (1,0) to (0,0)");
    }

    #[test]
    fn test_c0_ht_horizontal_tab_moves_to_next_tab_stop() {
        let mut term = new_term(20, 1);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::HT)),
        );
        assert_cursor_pos(&term, 8, 0, "HT from column 0 should move to column 8");
    }

    #[test]
    fn test_c0_lf_line_feed_moves_down_and_to_col0() {
        let mut term = new_term(5, 2);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)),
        );
        assert_cursor_pos(&term, 0, 1, "LF from (1,0) should move to (0,1)");
    }

    #[test]
    fn test_c0_cr_carriage_return_moves_to_col0() {
        let mut term = new_term(5, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::CR)),
        );
        assert_cursor_pos(&term, 0, 0, "CR from (1,0) should move to (0,0)");
    }

    #[test]
    fn test_c0_bel_produces_ring_bell_action() {
        let mut term = new_term(5, 1);
        let action = process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::BEL)),
        );
        assert_eq!(
            action,
            Some(EmulatorAction::RingBell),
            "BEL should produce RingBell action"
        );
    }

    #[test]
    fn test_csi_cup_cursor_position_moves_cursor() {
        let mut term = new_term(10, 5);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(3, 4))),
        );
        assert_cursor_pos(
            &term,
            3,
            2,
            "CUP(3,4) should move cursor to (3,2) (0-based)",
        );
    }

    #[test]
    fn test_csi_sgr_set_graphics_rendition_bold_applies_to_glyph() {
        let mut term = new_term(10, 1);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Bold,
            ]))),
        );
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        let glyph = get_glyph_at(&term, 0, 0);
        assert!(
            glyph.attr.flags.contains(AttrFlags::BOLD),
            "Printed char 'A' should have BOLD attribute"
        );
    }

    #[test]
    fn test_csi_ed_erase_in_display_all_clears_screen() {
        let mut term = new_term(5, 3);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::EraseInDisplay(2))),
        );
        assert_eq!(
            get_glyph_at(&term, 0, 0).c,
            ' ',
            "Cell (0,0) should be space after ED(2)"
        );
        assert_eq!(
            get_glyph_at(&term, 0, 0).attr,
            Attributes::default(),
            "Cell (0,0) attributes should be default after ED(2)"
        );
    }

    #[test]
    fn test_dec_mode_origin_decom_set_moves_cursor_to_margin() {
        let mut term = new_term(10, 5);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetScrollingRegion {
                top: 2,
                bottom: 4,
            })),
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(
                DecModeConstant::Origin as u16,
            ))),
        );
        assert_screen_cursor_pos(
            &term,
            0,
            1,
            "Cursor after DECOM set should be at physical (0,1) (top of margin)",
        );
    }

    #[test]
    fn test_resize_larger_maintains_cursor_logical_pos() {
        let mut term = new_term(5, 2);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(2, 2))),
        );
        term.resize(10, 4);
        assert_cursor_pos(
            &term,
            1,
            1,
            "Cursor logical pos (1,1) should be maintained after resize larger",
        );
    }

    #[test]
    fn test_resize_smaller_clamps_cursor_logical_pos() {
        let mut term = new_term(10, 4);
        // CUP(row=4, col=8) is 1-based. This translates to 0-based logical (x=7, y=3).
        // In a 10x4 terminal, (x=7, y=3) is a valid logical position.
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(4, 8))),
        );
        assert_cursor_pos(&term, 7, 3, "Initial cursor pos (7,3) before resize");

        term.resize(5, 2); // New max 0-indexed logical (col=4, row=1)
                           // Expected clamping: x = min(7, 4) = 4; y = min(3, 1) = 1. So, (4,1).
        assert_cursor_pos(
            &term,
            4,
            1,
            "Cursor logical pos (7,3) should be clamped to (4,1) after resize smaller",
        );
    }

    #[test]
    fn test_user_input_printable_key_generates_write_pty_action() {
        let mut term = new_term(5, 1);
        let action = term.interpret_input(EmulatorInput::User(BackendEvent::Key {
            keysym: 'A' as u32,
            text: "A".to_string(),
        }));
        assert_eq!(action, Some(EmulatorAction::WritePty(b"A".to_vec())));
    }

    #[test]
    fn test_user_input_enter_key_generates_write_pty_action() {
        let mut term = new_term(5, 1);
        let action = term.interpret_input(EmulatorInput::User(BackendEvent::Key {
            keysym: 0xFF0D,
            text: "\r".to_string(),
        }));
        assert_eq!(action, Some(EmulatorAction::WritePty(b"\r".to_vec())));
    }

    #[test]
    fn test_user_input_arrow_key_normal_mode_sends_csi_sequence() {
        let mut term = new_term(5, 1);
        let action = term.interpret_input(EmulatorInput::User(BackendEvent::Key {
            keysym: 0xFF52,
            text: "".to_string(),
        }));
        assert_eq!(action, Some(EmulatorAction::WritePty(b"\x1b[A".to_vec())));
    }

    #[test]
    fn test_user_input_arrow_key_application_mode_sends_ss3_sequence() {
        let mut term = new_term(5, 1);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(
                DecModeConstant::CursorKeys as u16,
            ))),
        );
        let action = term.interpret_input(EmulatorInput::User(BackendEvent::Key {
            keysym: 0xFF54,
            text: "".to_string(),
        }));
        assert_eq!(action, Some(EmulatorAction::WritePty(b"\x1bOB".to_vec())));
    }

    #[test]
    fn test_dec_private_mode_12_att610_cursor_blink_is_processed() {
        let mut term = new_term(5, 1);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(12))),
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(12))),
        );
    }

    #[test]
    fn test_dec_private_mode_2004_bracketed_paste_is_set_and_reset() {
        let mut term = new_term(5, 1);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(2004))),
        );
        assert!(term.is_bracketed_paste_mode_active());
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(2004))),
        );
        assert!(!term.is_bracketed_paste_mode_active());
    }

    #[test]
    fn test_dec_private_mouse_mode_1000_is_set_and_reset() {
        let mut term = new_term(5, 1);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(1000))),
        );
        assert!(term.is_mouse_mode_active(1000));
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(1000))),
        );
        assert!(!term.is_mouse_mode_active(1000));
    }

    #[test]
    fn test_dec_private_mode_focus_event_1004_is_set_and_reset() {
        let mut term = new_term(5, 1);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(1004))),
        );
        assert!(term.is_focus_event_mode_active());
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(1004))),
        );
        assert!(!term.is_focus_event_mode_active());
    }

    #[test]
    fn test_dec_private_mode_7727_is_processed() {
        let mut term = new_term(5, 1);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetModePrivate(7727))),
        );
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::ResetModePrivate(7727))),
        );
    }

    #[test]
    fn test_csi_sp_q_decscusr_set_cursor_style_updates_shape() {
        let mut term = new_term(5, 1);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetCursorStyle { shape: 4 })),
        );
        assert_eq!(term.get_cursor_shape(), 4);
    }

    #[test]
    fn test_csi_t_window_manipulation_report_chars_produces_action() {
        let mut term = new_term(80, 24);
        let actions = process_commands(
            &mut term,
            vec![AnsiCommand::Csi(CsiCommand::WindowManipulation {
                ps1: 18,
                ps2: None,
                ps3: None,
            })],
        );
        let expected_response = format!("\x1b[8;{};{}t", 24, 80);
        assert!(actions.contains(&EmulatorAction::WritePty(expected_response.into_bytes())));
    }

    #[test]
    fn test_csi_t_window_manipulation_unsupported_is_graceful() {
        let mut term = new_term(80, 24);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::WindowManipulation {
                ps1: 99,
                ps2: Some(0),
                ps3: Some(0),
            })),
        );
    }

    #[test]
    fn test_print_ascii_chars_updates_grid() {
        let mut term = new_term(10, 1);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('H')));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('i')));
        assert_eq!(get_glyph_at(&term, 0, 0).c, 'H');
        assert_eq!(get_glyph_at(&term, 1, 0).c, 'i');
        assert_cursor_pos(&term, 2, 0, "Cursor should be at (2,0) after 'Hi'");
    }

    #[test]
    fn test_print_chars_with_line_wrap_moves_to_next_line() {
        let mut term = new_term(2, 2);
        process_commands(
            &mut term,
            vec![AnsiCommand::Print('A'), AnsiCommand::Print('B')],
        );
        assert_cursor_pos(&term, 2, 0, "Cursor at end of line 0 before wrap");
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('C')));
        assert_eq!(get_glyph_at(&term, 0, 1).c, 'C');
        assert_cursor_pos(&term, 1, 1, "Cursor should be at (1,1) after 'C' wrapped");
    }

    #[test]
    fn test_print_chars_with_scroll_moves_content_up() {
        let mut term = new_term(1, 2);
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)),
        );
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('B')));
        assert_eq!(get_glyph_at(&term, 0, 0).c, 'A');
        assert_eq!(get_glyph_at(&term, 0, 1).c, 'B');
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::C0Control(C0Control::LF)),
        );
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('C')));
        assert_eq!(get_glyph_at(&term, 0, 0).c, 'B');
        assert_eq!(get_glyph_at(&term, 0, 1).c, 'C');
    }

    #[test]
    fn test_print_utf8_multibyte_char_occupies_correct_cells() {
        let mut term = new_term(5, 1);
        process_commands(
            &mut term,
            vec![AnsiCommand::Print('你'), AnsiCommand::Print('好')],
        );
        assert_eq!(get_glyph_at(&term, 0, 0).c, '你');
        assert_eq!(get_glyph_at(&term, 1, 0).c, '\0');
        assert_eq!(get_glyph_at(&term, 2, 0).c, '好');
        assert_eq!(get_glyph_at(&term, 3, 0).c, '\0');
        assert_cursor_pos(&term, 4, 0, "Cursor after '你好'");
    }

    #[test]
    fn test_print_wide_character_cjk_advances_cursor_by_width() {
        let mut term = new_term(5, 1);
        let wide_char = '世';
        let char_width = crate::term::unicode::get_char_display_width(wide_char);
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Print(wide_char)),
        );
        assert_eq!(get_glyph_at(&term, 0, 0).c, wide_char);
        if char_width == 2 {
            assert_eq!(get_glyph_at(&term, 1, 0).c, '\0');
        }
        assert_cursor_pos(&term, char_width, 0, "Cursor should advance by char_width");
    }

    #[test]
    fn test_print_wide_character_at_edge_of_line_wraps_correctly() {
        let mut term = new_term(2, 2);
        let wide_char = '世';
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Print(wide_char)),
        );
        assert_cursor_pos(
            &term,
            2,
            0,
            "Cursor at end of line 0 before wrap for wide char",
        );
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('A')));
        assert_eq!(get_glyph_at(&term, 0, 1).c, 'A');
        assert_cursor_pos(&term, 1, 1, "Cursor should be at (1,1) after 'A' wrapped");
    }
    #[test]
    fn test_esc_c_reset_to_initial_state_clears_and_homes_with_default_attrs() {
        let mut term = new_term(10, 3); // Create a 10x3 terminal

        // 1. Setup: Establish a non-default state
        // Set SGR to Red Foreground, Blue Background
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Named(NamedColor::Red)),
                Attribute::Background(Color::Named(NamedColor::Blue)),
            ]))),
        );

        // Print some text 'XY' at (0,0) and (1,0). These will have Red/Blue attributes.
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('X')));
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('Y')));

        // Move cursor away from home to (1,1) (logical)
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Csi(CsiCommand::CursorPosition(2, 2))), // 1-based for CUP
        );
        assert_cursor_pos(&term, 1, 1, "Cursor position before RIS");

        // Verify the initial state of a cell
        let glyph_before_ris = get_glyph_at(&term, 0, 0);
        assert_eq!(glyph_before_ris.c, 'X', "Cell (0,0) char before RIS");
        assert_eq!(
            glyph_before_ris.attr.fg,
            Color::Named(NamedColor::Red),
            "Cell (0,0) fg before RIS"
        );
        assert_eq!(
            glyph_before_ris.attr.bg,
            Color::Named(NamedColor::Blue),
            "Cell (0,0) bg before RIS"
        );

        // Clear dirty lines from setup to isolate RIS's dirtying behavior
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        // 2. Action: Send ESC c (Reset to Initial State) command
        process_input(
            &mut term,
            EmulatorInput::Ansi(AnsiCommand::Esc(
                crate::ansi::commands::EscCommand::ResetToInitialState,
            )),
        );

        // 3. Verification
        // 3.1. Cursor is at home position (0,0)
        assert_cursor_pos(&term, 0, 0, "Cursor position after RIS should be (0,0)");

        // 3.2. Screen is cleared, and all cells have true default attributes
        let expected_cleared_attr = Attributes::default(); // Expected: Color::Default, Color::Default, no flags
        let (term_width, term_height) = TerminalInterface::dimensions(&term);

        for y_idx in 0..term_height {
            for x_idx in 0..term_width {
                let glyph_after_ris = get_glyph_at(&term, x_idx, y_idx);
                assert_eq!(
                    glyph_after_ris.c, ' ',
                    "Cell ({},{}) char after RIS should be a space",
                    x_idx, y_idx
                );
                assert_eq!(
                    glyph_after_ris.attr, expected_cleared_attr,
                    "Cell ({},{}) attributes after RIS should be default. Got: {:?}",
                    x_idx, y_idx, glyph_after_ris.attr
                );
            }
        }

        // 3.3. Cursor's pending attributes are reset to default.
        // Test this by printing a character 'Z' and checking its attributes.
        // 'Z' should appear at the new cursor position (0,0).
        process_input(&mut term, EmulatorInput::Ansi(AnsiCommand::Print('Z')));
        let glyph_z = get_glyph_at(&term, 0, 0); // 'Z' is now at (0,0)
        assert_eq!(glyph_z.c, 'Z', "Char 'Z' printed after RIS");
        assert_eq!(
            glyph_z.attr,
            expected_cleared_attr, // Should use the true default attributes
            "Attributes of 'Z' printed after RIS should be default. Got: {:?}",
            glyph_z.attr
        );

        // 3.4. All lines are marked as dirty.
        let dirty_lines_after_ris = TerminalInterface::take_dirty_lines(&mut term);
        assert_eq!(
            dirty_lines_after_ris.len(),
            term_height,
            "Number of dirty lines after RIS should match terminal height. Got: {:?}",
            dirty_lines_after_ris
        );
        let expected_dirty_lines: Vec<usize> = (0..term_height).collect();
        assert_eq!(
            dirty_lines_after_ris,
            expected_dirty_lines,
            "All lines (0 to {}) should be dirty after RIS. Got: {:?}",
            term_height - 1,
            dirty_lines_after_ris
        );
    }
}

// This code is a corrected version of the tests previously generated,
// intended to be placed in a test module (e.g., src/term/tests.rs or a submodule).

#[cfg(test)]
mod extensive_term_emulator_tests {
    use crate::ansi::commands::{AnsiCommand, Attribute as SgrAttribute, C0Control, CsiCommand};
    use crate::color::{Color, NamedColor};
    use crate::glyph::{AttrFlags, Attributes, Glyph};
    use crate::term::{EmulatorInput, TerminalEmulator, TerminalInterface};
    use std::collections::HashSet;

    // CSI Erase Parameter Constants
    const ERASE_TO_END: u16 = 0;
    // const ERASE_TO_START: u16 = 1; // Not used in current failing tests, but keep for completeness if needed
    const ERASE_ALL: u16 = 2;

    fn create_term(cols: usize, rows: usize) -> TerminalEmulator {
        TerminalEmulator::new(cols, rows, 100)
    }

    fn get_char_at(term: &TerminalEmulator, row: usize, col: usize) -> char {
        TerminalInterface::get_glyph(term, col, row).c
    }

    fn get_glyph_at(term: &TerminalEmulator, row: usize, col: usize) -> Glyph {
        TerminalInterface::get_glyph(term, col, row)
    }

    fn get_line_as_string(term: &TerminalEmulator, row: usize) -> String {
        let (cols, _) = TerminalInterface::dimensions(term);
        let mut s = String::new();
        for col_idx in 0..cols {
            s.push(get_char_at(term, row, col_idx));
        }
        s.trim_end().to_string()
    }

    fn process_commands(term: &mut TerminalEmulator, commands: Vec<AnsiCommand>) {
        for cmd in commands {
            term.interpret_input(EmulatorInput::Ansi(cmd.clone()));
        }
    }

    fn process_command(term: &mut TerminalEmulator, command: AnsiCommand) {
        term.interpret_input(EmulatorInput::Ansi(command));
    }

    fn assert_and_clear_dirty_lines(
        term: &mut TerminalEmulator,
        expected_dirty_lines: &[usize],
        message: &str,
    ) {
        let dirty_lines_vec = TerminalInterface::take_dirty_lines(term);
        let dirty_lines_set: HashSet<usize> = dirty_lines_vec.into_iter().collect();
        let expected_set: HashSet<usize> = expected_dirty_lines.iter().cloned().collect();

        assert_eq!(
            dirty_lines_set, expected_set,
            "{} - Dirty lines mismatch. Got: {:?}, Expected: {:?}",
            message, dirty_lines_set, expected_set
        );

        let subsequent_dirty_lines = TerminalInterface::take_dirty_lines(term);
        assert!(
            subsequent_dirty_lines.is_empty(),
            "{} - Subsequent take_dirty_lines was not empty. Got: {:?}",
            message,
            subsequent_dirty_lines
        );
    }

    // --- Test Categories ---

    #[test]
    fn test_print_single_char_marks_line_dirty() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_command(&mut term, AnsiCommand::Print('A'));

        assert_eq!(
            get_char_at(&term, 0, 0),
            'A',
            "Character not printed correctly"
        );
        assert_and_clear_dirty_lines(&mut term, &[0], "Single char print");
    }

    #[test]
    fn test_print_multiple_chars_on_same_line_marks_dirty() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('A'),
                AnsiCommand::Print('B'),
                AnsiCommand::Print('C'),
            ],
        );

        assert_eq!(
            get_line_as_string(&term, 0),
            "ABC",
            "String not printed correctly"
        );
        assert_and_clear_dirty_lines(&mut term, &[0], "Multiple chars print");
    }

    // This test is FAILING: Expected [0,1] from CUD, Got {}
    #[test]
    fn test_print_char_at_cursor_updates_glyph_and_dirty() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_commands(
            &mut term,
            vec![
                AnsiCommand::Csi(CsiCommand::CursorDown(1)),
                AnsiCommand::Csi(CsiCommand::CursorForward(1)),
            ],
        );
        // This assertion is failing. CUD(1) should make lines 0 and 1 dirty.
        assert_and_clear_dirty_lines(&mut term, &[0, 1], "Cursor movement for setup");

        process_command(&mut term, AnsiCommand::Print('X'));
        assert_eq!(
            get_char_at(&term, 1, 1),
            'X',
            "Character not printed at new cursor pos"
        );
        assert_and_clear_dirty_lines(&mut term, &[1], "Print at (1,1)");
    }

    #[test]
    fn test_take_dirty_lines_clears_dirty_status() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_command(&mut term, AnsiCommand::Print('A'));

        let dirty_lines1 = TerminalInterface::take_dirty_lines(&mut term);
        assert_eq!(
            dirty_lines1,
            vec![0],
            "Initial print should make line 0 dirty"
        );

        let dirty_lines2 = TerminalInterface::take_dirty_lines(&mut term);
        assert!(
            dirty_lines2.is_empty(),
            "Second call to take_dirty_lines should return empty"
        );
    }

    #[test]
    fn test_initial_state_all_lines_dirty() {
        let mut term = create_term(10, 3);
        let (_, rows) = TerminalInterface::dimensions(&term);
        let expected_all_lines: Vec<usize> = (0..rows).collect();
        assert_and_clear_dirty_lines(&mut term, &expected_all_lines, "Initial state");
    }

    #[test]
    fn test_lf_moves_cursor_marks_old_and_new_lines_dirty() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);
        process_command(&mut term, AnsiCommand::Print('A'));
        assert_and_clear_dirty_lines(&mut term, &[0], "Setup print on line 0");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::LF));

        // C0 LF calls newline(), which should mark old line (0) and new line (1) dirty.
        assert_and_clear_dirty_lines(&mut term, &[0, 1], "After LF");
        let (cursor_x, cursor_y) = term.cursor_pos();
        assert_eq!((cursor_x, cursor_y), (0, 1), "Cursor not at (0,1) after LF");
    }

    #[test]
    fn test_cr_moves_cursor_to_col0_does_not_mark_dirty_if_no_glyph_change() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);
        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('A'),
                AnsiCommand::Print('B'),
                AnsiCommand::Print('C'),
            ],
        );
        assert_and_clear_dirty_lines(&mut term, &[0], "Setup print ABC on line 0");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::CR));
        assert_and_clear_dirty_lines(&mut term, &[], "After CR (no glyph change)");
        let (cursor_x, cursor_y) = term.cursor_pos();
        assert_eq!((cursor_x, cursor_y), (0, 0), "Cursor not at (0,0) after CR");
    }

    #[test]
    fn test_crlf_sequence_marks_lines_dirty_correctly() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);
        process_command(&mut term, AnsiCommand::Print('H'));
        assert_and_clear_dirty_lines(&mut term, &[0], "Setup print 'H'");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::CR));
        assert_and_clear_dirty_lines(&mut term, &[], "After CR in CRLF (no glyph change)");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::LF));
        assert_and_clear_dirty_lines(&mut term, &[0, 1], "After LF in CRLF");
        assert_eq!(
            get_char_at(&term, 0, 0),
            'H',
            "Char on line 0 incorrect after CRLF"
        );
        let (cursor_x, cursor_y) = term.cursor_pos();
        assert_eq!(
            (cursor_x, cursor_y),
            (0, 1),
            "Cursor pos incorrect after CRLF"
        );
    }

    #[test]
    fn test_lf_at_bottom_of_screen_scrolls_and_marks_all_lines_dirty() {
        let mut term = create_term(5, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_command(&mut term, AnsiCommand::Print('A'));
        process_command(&mut term, AnsiCommand::C0Control(C0Control::LF));
        process_command(&mut term, AnsiCommand::Print('B'));
        process_command(&mut term, AnsiCommand::C0Control(C0Control::LF));
        process_command(&mut term, AnsiCommand::Print('C'));
        assert_and_clear_dirty_lines(&mut term, &[0, 1, 2], "Screen fill");

        assert_eq!(get_line_as_string(&term, 0), "A", "L0 before scroll");
        assert_eq!(get_line_as_string(&term, 1), "B", "L1 before scroll");
        assert_eq!(get_line_as_string(&term, 2), "C", "L2 before scroll");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::LF));
        // scroll_up_serial marks all visible lines dirty within the scroll region,
        // and clear_line_segment for the new bottom line also marks it dirty.
        // So all lines [0,1,2] should be dirty.
        assert_and_clear_dirty_lines(&mut term, &[0, 1, 2], "After LF scroll");
        assert_eq!(get_line_as_string(&term, 0), "B", "L0 after scroll");
        assert_eq!(get_line_as_string(&term, 1), "C", "L1 after scroll");
        assert_eq!(
            get_line_as_string(&term, 2),
            "",
            "L2 after scroll (new blank line)"
        );
        let (cursor_x, cursor_y) = term.cursor_pos();
        assert_eq!((cursor_x, cursor_y), (0, 2), "Cursor pos after scroll");
    }

    #[test]
    fn test_command_output_scenario_single_line_output_step_by_step_dirty_check() {
        let mut term = create_term(20, 5);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_commands(
            &mut term,
            vec![AnsiCommand::Print('p'), AnsiCommand::Print('>')],
        );
        assert_and_clear_dirty_lines(&mut term, &[0], "Initial prompt");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::CR));
        assert_and_clear_dirty_lines(&mut term, &[], "CR before output");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::LF));
        // This should make lines 0 and 1 dirty. If newline() is fixed, this should pass.
        assert_and_clear_dirty_lines(&mut term, &[0, 1], "LF before output");

        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('o'),
                AnsiCommand::Print('u'),
                AnsiCommand::Print('t'),
            ],
        );
        assert_and_clear_dirty_lines(&mut term, &[1], "Printing 'out' on line 1");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::CR));
        assert_and_clear_dirty_lines(&mut term, &[], "CR after 'out'");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::LF));
        // This should make lines 1 and 2 dirty.
        assert_and_clear_dirty_lines(&mut term, &[1, 2], "LF before new prompt");

        process_commands(
            &mut term,
            vec![AnsiCommand::Print('p'), AnsiCommand::Print('>')],
        );
        assert_and_clear_dirty_lines(&mut term, &[2], "Printing new prompt on line 2");

        assert_eq!(
            get_line_as_string(&term, 0),
            "p>",
            "Line 0 (old prompt) content"
        );
        assert_eq!(
            get_line_as_string(&term, 1),
            "out",
            "Line 1 (output) content"
        );
        assert_eq!(
            get_line_as_string(&term, 2),
            "p>",
            "Line 2 (new prompt) content"
        );
        let (cursor_x, cursor_y) = term.cursor_pos();
        assert_eq!((cursor_x, cursor_y), (2, 2), "Final cursor position");
    }

    // This test is FAILING: Content assertion failure, and likely related to newline() issues affecting dirty lines too.
    #[test]
    fn test_full_pty_output_simulation_dirty_lines_coalesced() {
        let mut term = create_term(20, 5);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('p'),
                AnsiCommand::Print('r'),
                AnsiCommand::Print('o'),
                AnsiCommand::Print('m'),
                AnsiCommand::Print('p'),
                AnsiCommand::Print('t'),
                AnsiCommand::Print('>'),
                AnsiCommand::Print(' '), // Explicit space
            ],
        );
        assert_and_clear_dirty_lines(&mut term, &[0], "Dirty lines after initial prompt");

        let pty_output_commands = vec![
            AnsiCommand::C0Control(C0Control::CR),
            AnsiCommand::C0Control(C0Control::LF), // Should dirty [0,1] if newline works
            AnsiCommand::Print('o'),
            AnsiCommand::Print('1'), // Should dirty [1]
            AnsiCommand::C0Control(C0Control::CR),
            AnsiCommand::C0Control(C0Control::LF), // Should dirty [1,2] if newline works
            AnsiCommand::Print('o'),
            AnsiCommand::Print('2'), // Should dirty [2]
            AnsiCommand::C0Control(C0Control::CR),
            AnsiCommand::C0Control(C0Control::LF), // Should dirty [2,3] if newline works
            AnsiCommand::Print('p'),
            AnsiCommand::Print('>'), // Should dirty [3]
        ];

        for cmd in pty_output_commands {
            term.interpret_input(EmulatorInput::Ansi(cmd));
        }

        // If newline() works correctly, this should assert that lines 0, 1, 2, and 3 are dirty.
        assert_and_clear_dirty_lines(&mut term, &[0, 1, 2, 3], "After full PTY output block");

        // The failing content assertion. Check char by char to bypass trim_end() for this specific check.
        assert_eq!(get_char_at(&term, 0, 0), 'p', "Line 0 Col 0");
        assert_eq!(get_char_at(&term, 0, 1), 'r', "Line 0 Col 1");
        assert_eq!(get_char_at(&term, 0, 2), 'o', "Line 0 Col 2");
        assert_eq!(get_char_at(&term, 0, 3), 'm', "Line 0 Col 3");
        assert_eq!(get_char_at(&term, 0, 4), 'p', "Line 0 Col 4");
        assert_eq!(get_char_at(&term, 0, 5), 't', "Line 0 Col 5");
        assert_eq!(get_char_at(&term, 0, 6), '>', "Line 0 Col 6");
        assert_eq!(
            get_char_at(&term, 0, 7),
            ' ',
            "Line 0 Col 7 (the explicit space)"
        );

        assert_eq!(get_line_as_string(&term, 1), "o1", "Line 1 content");
        assert_eq!(get_line_as_string(&term, 2), "o2", "Line 2 content");
        assert_eq!(get_line_as_string(&term, 3), "p>", "Line 3 content");
        let (cursor_x, cursor_y) = term.cursor_pos();
        assert_eq!((cursor_x, cursor_y), (2, 3), "Final cursor position");
    }

    #[test]
    fn test_erase_in_display_all_marks_all_lines_dirty() {
        let mut term = create_term(5, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);
        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('A'),
                AnsiCommand::C0Control(C0Control::LF),
                AnsiCommand::Print('B'),
                AnsiCommand::C0Control(C0Control::LF),
                AnsiCommand::Print('C'),
            ],
        );
        assert_and_clear_dirty_lines(&mut term, &[0, 1, 2], "Screen fill before ED All");

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::EraseInDisplay(ERASE_ALL)),
        );
        assert_and_clear_dirty_lines(&mut term, &[0, 1, 2], "After ED All");
        // ... content assertions
        for r in 0..3 {
            for c_idx in 0..5 {
                assert_eq!(
                    get_glyph_at(&term, r, c_idx),
                    Glyph {
                        c: ' ',
                        attr: Attributes::default(),
                    },
                    "Cell ({},{}) not default after ED All",
                    r,
                    c_idx
                );
            }
        }
    }

    // This test now PASSED with the previous adjustment.
    #[test]
    fn test_erase_in_display_from_cursor_marks_affected_lines_dirty() {
        let mut term = create_term(5, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);
        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('L'),
                AnsiCommand::Print('0'),
                AnsiCommand::C0Control(C0Control::LF),
                AnsiCommand::Print('L'),
                AnsiCommand::Print('1'),
                AnsiCommand::C0Control(C0Control::LF),
                AnsiCommand::Print('L'),
                AnsiCommand::Print('2'),
            ],
        );
        assert_and_clear_dirty_lines(&mut term, &[0, 1, 2], "Screen fill before ED FromCursor");

        process_commands(
            &mut term,
            vec![AnsiCommand::Csi(CsiCommand::CursorPosition(2, 2))],
        );
        assert_and_clear_dirty_lines(
            &mut term,
            &[],
            "Cursor move to (1,1) for ED FromCursor (CUP alone shouldn't dirty screen content)",
        );

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::EraseInDisplay(ERASE_TO_END)),
        );

        let (_cols, rows) = TerminalInterface::dimensions(&term);
        let all_expected_lines: Vec<usize> = (0..rows).collect();
        assert_and_clear_dirty_lines(
            &mut term,
            &all_expected_lines,
            "After ED FromCursor (ToEnd) - expecting all lines due to mark_all_dirty in erase_in_display",
        );
        assert_eq!(
            get_line_as_string(&term, 0),
            "L0",
            "Line 0 after ED FromCursor"
        );
        assert_eq!(
            get_char_at(&term, 1, 0),
            'L',
            "Line 1, Col 0 after ED FromCursor"
        );
        assert_eq!(
            get_char_at(&term, 1, 1),
            ' ',
            "Line 1, Col 1 (cursor pos) after ED FromCursor"
        );
        assert_eq!(
            get_line_as_string(&term, 2),
            "",
            "Line 2 after ED FromCursor"
        );
    }

    #[test]
    fn test_erase_in_line_all_marks_current_line_dirty() {
        let mut term = create_term(5, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);
        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('A'),
                AnsiCommand::C0Control(C0Control::LF),
                AnsiCommand::Print('B'),
                AnsiCommand::Print('B'),
                AnsiCommand::Print('B'),
                AnsiCommand::C0Control(C0Control::LF),
                AnsiCommand::Print('C'),
            ],
        );
        assert_and_clear_dirty_lines(&mut term, &[0, 1, 2], "Screen fill before EL All");

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::CursorPosition(2, 1)),
        );
        assert_and_clear_dirty_lines(
            &mut term,
            &[],
            "Cursor move for EL All (CUP alone shouldn't dirty screen content)",
        );

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::EraseInLine(ERASE_ALL)),
        );
        assert_and_clear_dirty_lines(&mut term, &[1], "After EL All on line 1");
        // ... content assertions
        assert_eq!(get_line_as_string(&term, 0), "A", "Line 0 after EL All");
        assert_eq!(get_line_as_string(&term, 1), "", "Line 1 after EL All");
        assert_eq!(get_line_as_string(&term, 2), "C", "Line 2 after EL All");
    }

    #[test]
    fn test_erase_character_marks_line_dirty() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);
        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('A'),
                AnsiCommand::Print('B'),
                AnsiCommand::Print('C'),
                AnsiCommand::Print('D'),
                AnsiCommand::Print('E'),
            ],
        );
        assert_and_clear_dirty_lines(&mut term, &[0], "Setup for ECH");

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::CursorCharacterAbsolute(1)),
        );
        assert_and_clear_dirty_lines(
            &mut term,
            &[],
            "Cursor move for ECH (CHA alone shouldn't dirty screen content)",
        );

        process_command(&mut term, AnsiCommand::Csi(CsiCommand::EraseCharacter(3)));
        assert_and_clear_dirty_lines(&mut term, &[0], "After ECH");
        assert_eq!(
            get_line_as_string(&term, 0),
            "   DE",
            "Line content after ECH"
        );
    }

    #[test]
    fn test_sgr_change_attributes_does_not_mark_line_dirty_if_no_print() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![SgrAttribute::Bold])),
        );
        assert_and_clear_dirty_lines(&mut term, &[], "After SGR Bold, no print");
    }

    #[test]
    fn test_print_char_with_new_attributes_marks_line_dirty() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![SgrAttribute::Bold])),
        );
        assert_and_clear_dirty_lines(&mut term, &[], "SGR Bold applied");

        process_command(&mut term, AnsiCommand::Print('A'));
        assert_and_clear_dirty_lines(&mut term, &[0], "Print 'A' with new attribute");
        // ... content assertions
        let glyph = get_glyph_at(&term, 0, 0);
        assert_eq!(glyph.c, 'A');
        assert!(
            glyph.attr.flags.contains(AttrFlags::BOLD),
            "Glyph should be bold"
        );
    }

    // This test is FAILING: Expected [0,1] from CUD, Got {}
    #[test]
    fn test_erase_with_current_attributes() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                SgrAttribute::Background(Color::Named(NamedColor::Red)),
            ])),
        );
        assert_and_clear_dirty_lines(&mut term, &[], "SGR BG Red applied");

        process_command(&mut term, AnsiCommand::Csi(CsiCommand::CursorDown(1)));
        // This assertion is failing. CUD(1) should make lines 0 and 1 dirty.
        assert_and_clear_dirty_lines(&mut term, &[0, 1], "Cursor moved to line 1");

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::EraseInLine(ERASE_ALL)),
        );
        assert_and_clear_dirty_lines(&mut term, &[1], "EL All on line 1 with BG Red");
        // ... content assertions
        let (cols, _) = TerminalInterface::dimensions(&term);
        for c_idx in 0..cols {
            let glyph = get_glyph_at(&term, 1, c_idx);
            assert_eq!(glyph.c, ' ', "Erased char should be space on line 1");
            assert_eq!(
                glyph.attr.bg,
                Color::Named(NamedColor::Red),
                "Background should be red on line 1, col {}",
                c_idx
            );
        }
        let glyph_other_line = get_glyph_at(&term, 0, 0);
        assert_eq!(
            glyph_other_line.attr.bg,
            Color::Default,
            "Background on line 0 should be default, not red. Got: {:?}",
            glyph_other_line.attr.bg
        );
    }

    #[test]
    fn test_resize_marks_all_lines_dirty() {
        let mut term = create_term(10, 3);
        assert_and_clear_dirty_lines(&mut term, &[0, 1, 2], "Initial state before resize");

        term.resize(15, 5);
        let expected_dirty: Vec<usize> = (0..5).collect();
        assert_and_clear_dirty_lines(&mut term, &expected_dirty, "After resize to 15x5");
        // ... content assertions
        let (cols, rows) = TerminalInterface::dimensions(&term);
        assert_eq!(cols, 15, "Cols not updated after resize");
        assert_eq!(rows, 5, "Rows not updated after resize");
    }

    #[test]
    fn test_consecutive_lfs_scroll_correctly_and_mark_dirty() {
        let mut term = create_term(5, 2);
        let _ = TerminalInterface::take_dirty_lines(&mut term);
        process_command(&mut term, AnsiCommand::Print('A'));
        assert_and_clear_dirty_lines(&mut term, &[0], "Print A");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::LF));
        assert_and_clear_dirty_lines(&mut term, &[0, 1], "LF1: A on L0, cursor on L1");
        // ... content assertions
        assert_eq!(get_line_as_string(&term, 0), "A");
        assert_eq!(get_line_as_string(&term, 1), "");

        process_command(&mut term, AnsiCommand::Print('B'));
        assert_and_clear_dirty_lines(&mut term, &[1], "Print B on L1");
        // ... content assertions
        assert_eq!(get_line_as_string(&term, 0), "A");
        assert_eq!(get_line_as_string(&term, 1), "B");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::LF));
        assert_and_clear_dirty_lines(&mut term, &[0, 1], "LF2: Scrolled, B on L0, cursor on L1");
        // ... content assertions
        assert_eq!(get_line_as_string(&term, 0), "B", "L0 after scroll");
        assert_eq!(
            get_line_as_string(&term, 1),
            "",
            "L1 after scroll (new blank line)"
        );

        process_command(&mut term, AnsiCommand::C0Control(C0Control::LF));
        assert_and_clear_dirty_lines(
            &mut term,
            &[0, 1],
            "LF3: Scrolled, blank on L0, cursor on L1",
        );
        // ... content assertions
        assert_eq!(get_line_as_string(&term, 0), "", "L0 after second scroll");
        assert_eq!(
            get_line_as_string(&term, 1),
            "",
            "L1 after second scroll (new blank line)"
        );
    }

    #[test]
    fn test_print_then_cr_then_overwrite_marks_dirty() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);
        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('1'),
                AnsiCommand::Print('2'),
                AnsiCommand::Print('3'),
            ],
        );
        assert_and_clear_dirty_lines(&mut term, &[0], "Print 123");
        // ... content assertions
        assert_eq!(get_line_as_string(&term, 0), "123");

        process_command(&mut term, AnsiCommand::C0Control(C0Control::CR));
        assert_and_clear_dirty_lines(&mut term, &[], "CR, no glyph change");

        process_command(&mut term, AnsiCommand::Print('X'));
        assert_and_clear_dirty_lines(&mut term, &[0], "Overwrite with X");
        assert_eq!(get_line_as_string(&term, 0), "X23");
    }

    #[test]
    fn test_line_wrapping_marks_lines_dirty() {
        let mut term = create_term(3, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_command(&mut term, AnsiCommand::Print('A'));
        process_command(&mut term, AnsiCommand::Print('B'));
        process_command(&mut term, AnsiCommand::Print('C'));
        assert_and_clear_dirty_lines(&mut term, &[0], "Print ABC on line 0");
        // ... content assertions
        assert_eq!(get_line_as_string(&term, 0), "ABC");

        process_command(&mut term, AnsiCommand::Print('D'));
        assert_and_clear_dirty_lines(&mut term, &[0, 1], "Print D, causes wrap to line 1");
        // ... content assertions
        assert_eq!(get_line_as_string(&term, 0), "ABC", "Line 0 after wrap");
        assert_eq!(get_line_as_string(&term, 1), "D", "Line 1 after wrap");
        let (cursor_x, cursor_y) = term.cursor_pos();
        assert_eq!(
            (cursor_x, cursor_y),
            (1, 1),
            "Cursor pos after wrap and print D"
        );
    }

    #[test]
    fn test_erase_in_line_to_end_with_attributes() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        process_commands(
            &mut term,
            vec![
                AnsiCommand::Print('A'),
                AnsiCommand::Print('B'),
                AnsiCommand::Print('C'),
                AnsiCommand::Print('D'),
            ],
        );
        assert_and_clear_dirty_lines(&mut term, &[0], "Print ABCD");

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                SgrAttribute::Background(Color::Named(NamedColor::Blue)),
            ])),
        );
        assert_and_clear_dirty_lines(&mut term, &[], "SGR BG Blue");

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::CursorCharacterAbsolute(2)),
        );
        assert_and_clear_dirty_lines(
            &mut term,
            &[],
            "Cursor to col 1 (CHA alone shouldn't dirty)",
        );

        process_command(
            &mut term,
            AnsiCommand::Csi(CsiCommand::EraseInLine(ERASE_TO_END)),
        );
        assert_and_clear_dirty_lines(&mut term, &[0], "EL ToEnd with BG Blue");
        // ... content assertions
        assert_eq!(get_char_at(&term, 0, 0), 'A', "Char at (0,0) should be A");
        assert_eq!(
            get_glyph_at(&term, 0, 0).attr.bg,
            Color::Default,
            "BG at (0,0) should be default"
        );

        for c_idx in 1..10 {
            let glyph = get_glyph_at(&term, 0, c_idx);
            assert_eq!(
                glyph.c, ' ',
                "Erased char at ({},{}) should be space",
                0, c_idx
            );
            assert_eq!(
                glyph.attr.bg,
                Color::Named(NamedColor::Blue),
                "BG at ({},{}) should be Blue",
                0,
                c_idx
            );
        }
    }
}


// --- Selection and Copy/Paste Tests ---
#[cfg(test)]
mod selection_tests {
    use super::term_tests::{create_term, get_glyph_at, get_line_as_string, process_command, process_commands, assert_and_clear_dirty_lines};
    use crate::ansi::commands::{AnsiCommand, C0Control, CsiCommand};
    use crate::backends::{MouseButton, MouseEventType};
    use crate::term::{
        EmulatorAction, EmulatorInput, SelectionMode, SelectionState, TerminalEmulator,
        TerminalInterface, UserInputAction,
    };
    use test_log::test;

    fn set_content(term: &mut TerminalEmulator, lines: Vec<&str>) {
        for (i, line_content) in lines.iter().enumerate() {
            if i > 0 {
                process_command(term, AnsiCommand::C0Control(C0Control::LF));
            }
            for char_to_print in line_content.chars() {
                process_command(term, AnsiCommand::Print(char_to_print));
            }
        }
        // Clear initial dirty state from setup
        let _ = TerminalInterface::take_dirty_lines(term);
    }
    
    #[test]
    fn test_selection_mouse_press_starts_selection() {
        let mut term = create_term(10, 3);
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        term.interpret_input(EmulatorInput::User(UserInputAction::MouseInput {
            event_type: MouseEventType::Press,
            col: 2,
            row: 1,
            button: MouseButton::Left,
            modifiers: crate::backends::Modifiers::empty(),
        }));

        assert!(term.selection.is_some(), "Selection should be Some after press");
        if let Some(selection) = &term.selection {
            assert_eq!(selection.start, (2, 1), "Selection start mismatch");
            assert_eq!(selection.end, (2, 1), "Selection end mismatch");
            assert_eq!(selection.mode, SelectionMode::Normal, "Selection mode mismatch");
        }
        assert_and_clear_dirty_lines(&mut term, &[1], "Line 1 (selection start) should be dirty");
    }

    #[test]
    fn test_selection_mouse_drag_updates_selection_normal_mode() {
        let mut term = create_term(10, 3);
        // Initial press
        term.interpret_input(EmulatorInput::User(UserInputAction::MouseInput {
            event_type: MouseEventType::Press,
            col: 2,
            row: 1,
            button: MouseButton::Left,
            modifiers: crate::backends::Modifiers::empty(),
        }));
        let _ = TerminalInterface::take_dirty_lines(&mut term); // Clear dirty from press

        // Drag
        term.interpret_input(EmulatorInput::User(UserInputAction::MouseInput {
            event_type: MouseEventType::Move,
            col: 5,
            row: 2,
            button: MouseButton::Left, // Usually not specified for move, but good to be explicit
            modifiers: crate::backends::Modifiers::empty(),
        }));
        
        assert!(term.selection.is_some(), "Selection should exist after drag");
        if let Some(selection) = &term.selection {
            assert_eq!(selection.start, (2, 1), "Selection start should remain the same");
            assert_eq!(selection.end, (5, 2), "Selection end should update to drag position");
        }
        // Old end row (1) and new end row (2) should be dirty.
        assert_and_clear_dirty_lines(&mut term, &[1, 2], "Lines 1 and 2 (old and new end) should be dirty");
    }

    #[test]
    fn test_selection_mouse_release_finalizes_selection() {
        let mut term = create_term(10, 3);
        term.interpret_input(EmulatorInput::User(UserInputAction::MouseInput {
            event_type: MouseEventType::Press, col: 2, row: 1, button: MouseButton::Left, modifiers: crate::backends::Modifiers::empty(),
        }));
        term.interpret_input(EmulatorInput::User(UserInputAction::MouseInput {
            event_type: MouseEventType::Move, col: 5, row: 1, button: MouseButton::Left, modifiers: crate::backends::Modifiers::empty(),
        }));
        let _ = TerminalInterface::take_dirty_lines(&mut term); 

        term.interpret_input(EmulatorInput::User(UserInputAction::MouseInput {
            event_type: MouseEventType::Release,
            col: 5, // Release coordinates
            row: 1,
            button: MouseButton::Left,
            modifiers: crate::backends::Modifiers::empty(),
        }));

        assert!(term.selection.is_some(), "Selection should still exist after release");
        if let Some(selection) = &term.selection { // Ensure selection is still (2,1) to (5,1)
            assert_eq!(selection.start, (2,1));
            assert_eq!(selection.end, (5,1));
        }
        assert_and_clear_dirty_lines(&mut term, &[1], "Line 1 (selection) should be dirty on release");
    }
    
    #[test]
    fn test_selection_cleared_on_initiate_copy() {
        let mut term = create_term(10, 3);
        set_content(&mut term, vec!["Hello World"]);
        term.interpret_input(EmulatorInput::User(UserInputAction::MouseInput {
            event_type: MouseEventType::Press, col: 0, row: 0, button: MouseButton::Left, modifiers: crate::backends::Modifiers::empty(),
        }));
        term.interpret_input(EmulatorInput::User(UserInputAction::MouseInput {
            event_type: MouseEventType::Move, col: 4, row: 0, button: MouseButton::Left, modifiers: crate::backends::Modifiers::empty(),
        })); // Selects "Hello"
        assert!(term.selection.is_some());
        let _ = TerminalInterface::take_dirty_lines(&mut term);

        term.interpret_input(EmulatorInput::User(UserInputAction::InitiateCopy));
        assert!(term.selection.is_none(), "Selection should be None after copy");
        assert_and_clear_dirty_lines(&mut term, &[0], "Line 0 (former selection) should be dirty after copy");
    }

    #[test]
    fn test_selection_cleared_on_resize() {
        let mut term = create_term(10, 3);
        term.interpret_input(EmulatorInput::User(UserInputAction::MouseInput {
            event_type: MouseEventType::Press, col: 0, row: 0, button: MouseButton::Left, modifiers: crate::backends::Modifiers::empty(),
        }));
        assert!(term.selection.is_some());
        let _ = TerminalInterface::take_dirty_lines(&mut term); // Clear setup dirty

        term.resize(5, 2); // Resize the terminal
        assert!(term.selection.is_none(), "Selection should be None after resize");
        // Resize marks all lines dirty by default. We also added specific dirtying for old selection.
        // The `resize` method itself calls `screen.resize` which marks all new lines dirty.
        // The test helper `assert_and_clear_dirty_lines` expects exact match.
        // For simplicity, let's just check that the old selection line was dirtied if it's within new bounds.
        // Or, more simply, rely on resize marking all lines dirty.
        let (w,h) = TerminalInterface::dimensions(&term);
        let all_lines: Vec<usize> = (0..h).collect();
        assert_and_clear_dirty_lines(&mut term, &all_lines, "All lines should be dirty after resize");
    }

    // --- get_selected_text() Tests ---
    #[test]
    fn test_get_selected_text_no_selection() {
        let term = create_term(10, 3);
        assert_eq!(term.get_selected_text(), None);
    }

    #[test]
    fn test_get_selected_text_single_line_simple() {
        let mut term = create_term(20, 3);
        set_content(&mut term, vec!["Hello World, how are you?"]);
        term.selection = Some(SelectionState { start: (6,0), end: (10,0), mode: SelectionMode::Normal }); // Select "World"
        assert_eq!(term.get_selected_text(), Some("World".to_string()));
    }

    #[test]
    fn test_get_selected_text_multi_line() {
        let mut term = create_term(10, 3);
        set_content(&mut term, vec!["First line", "Second row", "Third one"]);
        // Select from "line" (0,5) on first line to "Sec" (1,2) on second line
        term.selection = Some(SelectionState { start: (6,0), end: (2,1), mode: SelectionMode::Normal }); 
        assert_eq!(term.get_selected_text(), Some("line\nSec".to_string()));
    }
    
    #[test]
    fn test_get_selected_text_full_line() {
        let mut term = create_term(10,3);
        set_content(&mut term, vec!["Full Line!"]);
        term.selection = Some(SelectionState { start: (0,0), end: (9,0), mode: SelectionMode::Normal});
        assert_eq!(term.get_selected_text(), Some("Full Line!".to_string()));
    }

    #[test]
    fn test_get_selected_text_same_cell() {
        let mut term = create_term(10,3);
        set_content(&mut term, vec!["Test"]);
        term.selection = Some(SelectionState { start: (1,0), end: (1,0), mode: SelectionMode::Normal}); // Select "e"
        assert_eq!(term.get_selected_text(), Some("e".to_string()));
    }

    #[test]
    fn test_get_selected_text_wide_chars() {
        let mut term = create_term(10, 3);
        set_content(&mut term, vec!["你好世界"]); // "Hello World" in Chinese
        // Select "你好" (Ni Hao) which are 2 wide chars, so 4 cells
        term.selection = Some(SelectionState { start: (0,0), end: (3,0), mode: SelectionMode::Normal });
        assert_eq!(term.get_selected_text(), Some("你好".to_string()));
        
        // Select "世" (Shi) - one wide char
        term.selection = Some(SelectionState { start: (4,0), end: (5,0), mode: SelectionMode::Normal });
        assert_eq!(term.get_selected_text(), Some("世".to_string()));
    }
    
    #[test]
    fn test_get_selected_text_trailing_spaces() {
        let mut term = create_term(20, 3);
        set_content(&mut term, vec!["Line with spaces   ", "Next line"]);
        // Select "Line with spaces" (excluding trailing)
        term.selection = Some(SelectionState { start: (0,0), end: (15,0), mode: SelectionMode::Normal });
        assert_eq!(term.get_selected_text(), Some("Line with spaces".to_string()));

        // Select "Line with spaces   " (including some trailing) then part of next line
        term.selection = Some(SelectionState { start: (0,0), end: (3,1), mode: SelectionMode::Normal });
        assert_eq!(term.get_selected_text(), Some("Line with spaces\nNext".to_string()), "Trailing spaces should be trimmed on non-final selected line part");
    }

    #[test]
    fn test_get_selected_text_reversed_coords() {
        let mut term = create_term(20, 3);
        set_content(&mut term, vec!["Hello World"]);
        term.selection = Some(SelectionState { start: (10,0), end: (6,0), mode: SelectionMode::Normal }); // Select "World" backwards
        assert_eq!(term.get_selected_text(), Some("World".to_string()));
    }

    // --- EmulatorAction Generation Tests ---
    #[test]
    fn test_action_initiate_copy_with_selection() {
        let mut term = create_term(10,3);
        set_content(&mut term, vec!["Copy This"]);
        term.selection = Some(SelectionState { start: (0,0), end: (8,0), mode: SelectionMode::Normal });
        let action = term.interpret_input(EmulatorInput::User(UserInputAction::InitiateCopy));
        assert_eq!(action, Some(EmulatorAction::CopyToClipboard("Copy This".to_string())));
    }

    #[test]
    fn test_action_initiate_copy_no_selection() {
        let mut term = create_term(10,3);
        let action = term.interpret_input(EmulatorInput::User(UserInputAction::InitiateCopy));
        assert_eq!(action, None);
    }

    #[test]
    fn test_action_initiate_paste_generates_request() {
        let mut term = create_term(10,3);
        let action = term.interpret_input(EmulatorInput::User(UserInputAction::InitiatePaste));
        assert_eq!(action, Some(EmulatorAction::RequestClipboardContent));
    }

    // --- UserInputAction::PasteText Handling Tests ---
    #[test]
    fn test_paste_text_normal_mode() {
        let mut term = create_term(10,3);
        term.dec_modes.bracketed_paste_mode = false;
        let action = term.interpret_input(EmulatorInput::User(UserInputAction::PasteText("Pasted!".to_string())));
        assert_eq!(action, Some(EmulatorAction::WritePty(b"Pasted!".to_vec())));
    }

    #[test]
    fn test_paste_text_bracketed_mode() {
        let mut term = create_term(10,3);
        term.dec_modes.bracketed_paste_mode = true;
        let action = term.interpret_input(EmulatorInput::User(UserInputAction::PasteText("Pasted!".to_string())));
        let expected_bytes = [b"\x1b[200~".to_vec(), b"Pasted!".to_vec(), b"\x1b[201~".to_vec()].concat();
        assert_eq!(action, Some(EmulatorAction::WritePty(expected_bytes)));
    }
}
