// src/term/tests.rs

//! Unit tests for the main TerminalEmulator struct and its core logic.
//! These tests aim to verify the internal state changes and actions produced
//! by the TerminalEmulator based on various inputs, adhering to its public API.

// Module for tests related to the Term struct itself
#[cfg(test)]
mod term_tests {
    use crate::ansi::commands::{
        AnsiCommand, Attribute, C0Control, Color as AnsiColor, CsiCommand,
    };
    use crate::backends::BackendEvent;
    use crate::glyph::{AttrFlags, Attributes, Color, DEFAULT_GLYPH, Glyph, NamedColor};
    use crate::term::{
        DecModeConstant,
        EmulatorAction,
        EmulatorInput,
        TerminalEmulator,
        TerminalInterface, // Import the trait to bring its methods into scope
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
                Attribute::Foreground(AnsiColor::Red),
                Attribute::Background(AnsiColor::Blue),
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
                Attribute::Foreground(AnsiColor::Red),
                Attribute::Background(AnsiColor::Blue),
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
                Attribute::Foreground(AnsiColor::Red),
                Attribute::Background(AnsiColor::Blue),
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
        let expected_attrs = DEFAULT_GLYPH.attr; // Compare with the constant

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
                Attribute::Foreground(AnsiColor::Green),
                Attribute::Background(AnsiColor::Magenta),
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
                Attribute::Foreground(AnsiColor::Cyan),
                Attribute::Background(AnsiColor::Yellow),
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
            DEFAULT_GLYPH.attr,
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
                Attribute::Foreground(AnsiColor::Red),
                Attribute::Background(AnsiColor::Blue),
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
            glyph_z.attr, expected_cleared_attr, // Should use the true default attributes
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
            dirty_lines_after_ris, expected_dirty_lines,
            "All lines (0 to {}) should be dirty after RIS. Got: {:?}",
            term_height - 1, dirty_lines_after_ris
        );
    }

}
