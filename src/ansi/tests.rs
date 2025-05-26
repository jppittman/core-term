// src/ansi/tests.rs

//! Tests for the ANSI parser and lexer integration.

// Use the AnsiProcessor, which combines lexer and parser
// Corrected imports using commands submodule path
use super::{
    AnsiParser, AnsiProcessor,
    commands::{AnsiCommand, Attribute, C0Control, CsiCommand, EscCommand},
};
use crate::color::{Color, NamedColor};
use test_log::test; // Ensure test_log is a dev-dependency for log capturing in tests

// Helper function to process bytes and get commands
fn process_bytes(bytes: &[u8]) -> Vec<AnsiCommand> {
    let mut processor = AnsiProcessor::new();
    processor.process_bytes(bytes)
}

#[test]
fn it_should_process_a_simple_printable_string() {
    let bytes = b"Hello, world!";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::Print('H'),
            AnsiCommand::Print('e'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('o'),
            AnsiCommand::Print(','),
            AnsiCommand::Print(' '),
            AnsiCommand::Print('w'),
            AnsiCommand::Print('o'),
            AnsiCommand::Print('r'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('d'),
            AnsiCommand::Print('!'),
        ]
    );
}

#[test]
fn it_should_process_c0_bel() {
    let bytes = b"\x07"; // BEL
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::BEL)]);
}

#[test]
fn it_should_process_csi_h_as_cup_1_1() {
    let bytes = b"\x1B[H"; // CSI H -> CUP (1, 1)
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::CursorPosition(1, 1))]
    );
}

#[test]
fn it_should_process_csi_sgr_reset() {
    let bytes = b"\x1B[0m";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Reset
        ]))]
    );
}

#[test]
fn it_should_process_csi_sgr_set_foreground() {
    let bytes = b"\x1B[34m";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Foreground(Color::Named(NamedColor::Blue))
        ]))]
    );
}

#[test]
fn it_should_process_dec_private_mode_reset_12_att610_cursor_blink() {
    let bytes = b"\x1b[?12l";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(12))],
        "Expected ResetModePrivate(12) for CSI ?12l"
    );
}

#[test]
fn it_should_process_dec_private_mode_set_25_text_cursor_enable() {
    let bytes = b"\x1b[?25h";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(25))],
        "Expected SetModePrivate(25) for CSI ?25h"
    );
}

#[test]
fn it_should_process_dec_private_mode_reset_25_text_cursor_enable() {
    let bytes = b"\x1b[?25l";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(25))],
        "Expected ResetModePrivate(25) for CSI ?25l"
    );
}

#[test]
fn it_should_process_various_dec_private_mouse_modes() {
    let modes_to_test = vec![
        (1000, b"\x1b[?1000h", b"\x1b[?1000l", "XTERM_MOUSE_CLICK"),
        (
            1002,
            b"\x1b[?1002h",
            b"\x1b[?1002l",
            "XTERM_MOUSE_BTN_MOTION",
        ),
        (
            1003,
            b"\x1b[?1003h",
            b"\x1b[?1003l",
            "XTERM_MOUSE_ANY_MOTION",
        ),
        (1005, b"\x1b[?1005h", b"\x1b[?1005l", "XTERM_MOUSE_UTF8"),
        (1006, b"\x1b[?1006h", b"\x1b[?1006l", "XTERM_MOUSE_SGR"),
    ];
    for (mode_num, set_seq, reset_seq, _name) in modes_to_test {
        let set_commands = process_bytes(set_seq);
        assert_eq!(
            set_commands,
            vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(mode_num))],
            "Expected SetModePrivate({}) for {:?}",
            mode_num,
            String::from_utf8_lossy(set_seq)
        );
        let reset_commands = process_bytes(reset_seq);
        assert_eq!(
            reset_commands,
            vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(mode_num))],
            "Expected ResetModePrivate({}) for {:?}",
            mode_num,
            String::from_utf8_lossy(reset_seq)
        );
    }
}

#[test]
fn it_should_process_dec_private_mode_bracketed_paste_2004() {
    let bytes_set = b"\x1b[?2004h";
    let commands_set = process_bytes(bytes_set);
    assert_eq!(
        commands_set,
        vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(2004))],
        "Expected SetModePrivate(2004) for CSI ?2004h"
    );

    let bytes_reset = b"\x1b[?2004l";
    let commands_reset = process_bytes(bytes_reset);
    assert_eq!(
        commands_reset,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(2004))],
        "Expected ResetModePrivate(2004) for CSI ?2004l"
    );
}

#[test]
fn it_should_process_dec_private_mode_focus_event_1004() {
    let bytes_set = b"\x1b[?1004h";
    let commands_set = process_bytes(bytes_set);
    assert_eq!(
        commands_set,
        vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(1004))],
        "Expected SetModePrivate(1004) for CSI ?1004h"
    );

    let bytes_reset = b"\x1b[?1004l";
    let commands_reset = process_bytes(bytes_reset);
    assert_eq!(
        commands_reset,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(1004))],
        "Expected ResetModePrivate(1004) for CSI ?1004l"
    );
}

#[test]
fn it_should_process_dec_private_mode_uncommon_7727() {
    let bytes = b"\x1b[?7727l";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(7727))],
        "Expected ResetModePrivate(7727) for CSI ?7727l"
    );
}

#[test]
fn it_should_process_csi_set_cursor_style_decscusr() {
    let bytes_steady_block = b"\x1b[2 q";
    let commands_steady_block = process_bytes(bytes_steady_block);
    assert_eq!(
        commands_steady_block,
        vec![AnsiCommand::Csi(CsiCommand::SetCursorStyle { shape: 2 })],
        "Expected SetCursorStyle for CSI 2 SP q"
    );

    let bytes_default_cursor = b"\x1b[0 q";
    let commands_default_cursor = process_bytes(bytes_default_cursor);
    assert_eq!(
        commands_default_cursor,
        vec![AnsiCommand::Csi(CsiCommand::SetCursorStyle { shape: 0 })],
        "Expected SetCursorStyle for CSI 0 SP q"
    );

    let bytes_blink_underline = b"\x1b[3 q";
    let commands_blink_underline = process_bytes(bytes_blink_underline);
    assert_eq!(
        commands_blink_underline,
        vec![AnsiCommand::Csi(CsiCommand::SetCursorStyle { shape: 3 })],
        "Expected SetCursorStyle for CSI 3 SP q"
    );
}

#[test]
fn it_should_process_csi_window_manipulation_t() {
    let bytes_23_0_0_t = b"\x1b[23;0;0t";
    let commands_23_0_0_t = process_bytes(bytes_23_0_0_t);
    assert_eq!(
        commands_23_0_0_t,
        vec![AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 23,
            ps2: Some(0),
            ps3: Some(0)
        })],
        "Expected WindowManipulation for CSI 23;0;0t"
    );

    let bytes_18_t = b"\x1b[18t";
    let commands_18_t = process_bytes(bytes_18_t);
    assert_eq!(
        commands_18_t,
        vec![AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 18,
            ps2: None,
            ps3: None
        })],
        "Expected WindowManipulation for CSI 18t"
    );

    let bytes_14_t = b"\x1b[14t";
    let commands_14_t = process_bytes(bytes_14_t);
    assert_eq!(
        commands_14_t,
        vec![AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 14,
            ps2: None,
            ps3: None
        })],
        "Expected WindowManipulation for CSI 14t"
    );
}

#[test]
fn it_should_process_csi_sequence_fragmented_across_param_bytes() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1B[1");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC [ 1)");
    let commands_frag2 = processor.process_bytes(b";2H");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Csi(CsiCommand::CursorPosition(1, 2))],
        "After fragment 2 (;2H)"
    );
}

#[test]
fn it_should_process_csi_sequence_fragmented_across_intermediate_bytes() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1B[?");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC [ ?)");
    let commands_frag2 = processor.process_bytes(b"25h");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(25))],
        "After fragment 2 (25h)"
    );
}

#[test]
fn it_should_process_csi_sequence_fragmented_after_esc() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1B");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC)");
    let commands_frag2 = processor.process_bytes(b"[1A");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Csi(CsiCommand::CursorUp(1))],
        "After fragment 2 ([1A)"
    );
}

#[test]
fn it_should_process_string_interspersed_with_fragmented_csi() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"Hello ");
    assert_eq!(
        commands_frag1,
        vec![
            AnsiCommand::Print('H'),
            AnsiCommand::Print('e'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('o'),
            AnsiCommand::Print(' '),
        ],
        "After fragment 1 (Hello )"
    );
    let commands_frag2 = processor.process_bytes(b"\x1B[31");
    assert_eq!(commands_frag2, vec![], "After fragment 2 (ESC [ 31)");
    let commands_frag3 = processor.process_bytes(b"m World");
    assert_eq!(
        commands_frag3,
        vec![
            AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
                Attribute::Foreground(Color::Named(NamedColor::Red))
            ])),
            AnsiCommand::Print(' '),
            AnsiCommand::Print('W'),
            AnsiCommand::Print('o'),
            AnsiCommand::Print('r'),
            AnsiCommand::Print('l'),
            AnsiCommand::Print('d'),
        ],
        "After fragment 3 (m World)"
    );
}

#[test]
fn it_should_handle_fragmented_utf8_input_with_intermediate_finalization() {
    // This test demonstrates how AnsiProcessor (which calls lexer.finalize()
    // within its process_bytes) handles UTF-8 fragments delivered in separate calls.
    let mut processor_refined = AnsiProcessor::new();
    assert_eq!(
        processor_refined.process_bytes(b"A"),
        vec![AnsiCommand::Print('A')],
        "Refined Frag 0: Print 'A'"
    );
    // \xE4 is start of '你'. Since it's an incomplete sequence when process_bytes finishes, finalize() converts it.
    assert_eq!(
        processor_refined.process_bytes(b"\xE4"),
        vec![AnsiCommand::Print(char::REPLACEMENT_CHARACTER)],
        "Refined Frag 1: Incomplete UTF-8 (E4) yields replacement char"
    );
    // \xBD is now treated as a new byte. It's an invalid UTF-8 start. finalize() converts it.
    assert_eq!(
        processor_refined.process_bytes(b"\xBD"),
        vec![AnsiCommand::Print(char::REPLACEMENT_CHARACTER)],
        "Refined Frag 2: Invalid UTF-8 start (BD) yields replacement char"
    );
    // \xA0 is also an invalid UTF-8 start. finalize() converts it.
    assert_eq!(
        processor_refined.process_bytes(b"\xA0"),
        vec![AnsiCommand::Print(char::REPLACEMENT_CHARACTER)],
        "Refined Frag 3: Invalid UTF-8 start (A0) yields replacement char"
    );
    assert_eq!(
        processor_refined.process_bytes(b"B"),
        vec![AnsiCommand::Print('B')],
        "Refined Frag 4: Print 'B'"
    );

    // For contrast, show how a complete multi-byte char is processed in one call
    let mut processor_complete = AnsiProcessor::new();
    assert_eq!(
        processor_complete.process_bytes(b"\xE4\xBD\xA0"),
        vec![AnsiCommand::Print('你')],
        "Complete '你' in one call"
    );
}

#[test]
fn it_should_complete_csi_if_final_byte_arrives_after_params() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1B[31");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC [ 31)");
    let commands_frag2 = processor.process_bytes(b"A");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Csi(CsiCommand::CursorUp(31))],
        "After fragment 2 (A)"
    );
    let commands_frag3 = processor.process_bytes(b"BC");
    assert_eq!(
        commands_frag3,
        vec![AnsiCommand::Print('B'), AnsiCommand::Print('C')],
        "After fragment 3 (BC)"
    );
}

#[test]
fn it_should_complete_osc_if_terminator_arrives_after_string_fragment() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1B]0;Ti");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC ] 0 ; Ti)");
    let commands_frag2 = processor.process_bytes(b"tle\x07");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Osc(b"0;Title".to_vec())],
        "After fragment 2 (tle BEL)"
    );
}

#[test]
fn it_should_complete_dcs_if_terminator_arrives_after_string_fragment() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"\x1BPSt");
    assert_eq!(commands_frag1, vec![], "After fragment 1 (ESC P St)");
    let commands_frag2 = processor.process_bytes(b"uff\x1B\\");
    assert_eq!(
        commands_frag2,
        vec![AnsiCommand::Dcs(b"Stuff".to_vec())],
        "After fragment 2 (uff ESC \\)"
    );
}

// --- String Sequence Tests ---

#[test]
fn it_should_process_osc_string_terminated_by_bel() {
    let bytes = b"\x1B]0;Set Title\x07";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Osc(b"0;Set Title".to_vec())]);
}

#[test]
fn it_should_process_osc_string_terminated_by_st() {
    let bytes = b"\x1B]2;Another Title\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Osc(b"2;Another Title".to_vec())]
    );
}

#[test]
fn it_should_process_dcs_string_terminated_by_st() {
    let bytes = b"\x1BP1;1$rText\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Dcs(b"1;1$rText".to_vec())]);
}

#[test]
fn it_should_process_pm_string_terminated_by_st() {
    let bytes = b"\x1B^Privacy Message\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Pm(b"Privacy Message".to_vec())]);
}

#[test]
fn it_should_process_apc_string_terminated_by_st() {
    let bytes = b"\x1B_Application Command\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Apc(b"Application Command".to_vec())]
    );
}

// --- Edge Case / Error Tests ---

#[test]
fn it_should_process_empty_input_as_no_commands() {
    let bytes = b"";
    let commands = process_bytes(bytes);
    assert!(commands.is_empty());
}

#[test]
fn it_should_buffer_incomplete_csi_sequence() {
    let bytes = b"\x1B[1;2"; // Incomplete CSI
    let commands = process_bytes(bytes);
    // AnsiProcessor calls finalize, which might clear incomplete CSI state
    // or the parser might hold it. If it holds, empty is correct.
    // If finalize clears it without error, empty is also correct.
    assert!(
        commands.is_empty(),
        "Incomplete CSI should not produce commands yet, or should be cleared by finalize if unrecoverable by next process_bytes call"
    );
}

#[test]
fn it_should_process_csi_with_invalid_final_byte_as_error() {
    let bytes = b"\x1B[31a"; // 'a' is not a valid CSI final byte
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Error(b'a')]);
}

#[test]
fn it_should_buffer_incomplete_osc_sequence() {
    let bytes = b"\x1B]0;Title"; // Incomplete OSC
    let commands = process_bytes(bytes);
    assert!(
        commands.is_empty(),
        "Incomplete OSC should not produce commands yet"
    );
}

#[test]
fn it_should_buffer_incomplete_dcs_sequence() {
    let bytes = b"\x1BPStuff"; // Incomplete DCS
    let commands = process_bytes(bytes);
    assert!(
        commands.is_empty(),
        "Incomplete DCS should not produce commands yet"
    );
}

#[test]
fn it_should_terminate_osc_on_bel_and_process_subsequent_chars() {
    let bytes = b"\x1B]0;String\x08with\x07BEL"; // BEL (0x07) terminates OSC
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::Osc(b"0;String\x08with".to_vec()),
            AnsiCommand::Print('B'),
            AnsiCommand::Print('E'),
            AnsiCommand::Print('L'),
        ]
    );
}

#[test]
fn it_should_abort_osc_on_esc_and_process_subsequent_commands() {
    let bytes = b"\x1B]0;String\x1B\x07BEL"; // ESC (0x1B) aborts OSC
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::C0Control(C0Control::ESC),
            AnsiCommand::C0Control(C0Control::BEL),
            AnsiCommand::Print('B'),
            AnsiCommand::Print('E'),
            AnsiCommand::Print('L'),
        ]
    );
}

#[test]
fn it_should_include_c0_controls_within_dcs_data() {
    let bytes = b"\x1BPString\x08with\x0BC0\x1B\\"; // C0 controls are part of DCS data
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Dcs(b"String\x08with\x0BC0".to_vec())]
    );
}

#[test]
fn it_should_abort_dcs_on_esc_and_process_subsequent_st() {
    let bytes = b"\x1BPString\x1B\x1B\\"; // First ESC aborts DCS, second ESC + \ is ST
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::C0Control(C0Control::ESC),
            AnsiCommand::StringTerminator,
        ]
    );
}

#[test]
fn it_should_process_st_in_ground_state() {
    let bytes_esc_st = b"\x1B\\"; // ST (ESC \)
    let commands_esc_st = process_bytes(bytes_esc_st);
    assert_eq!(commands_esc_st, vec![AnsiCommand::StringTerminator]);

    let bytes_c1_st = b"\x9C"; // ST (C1 version)
    let commands_c1_st = process_bytes(bytes_c1_st);
    assert_eq!(commands_c1_st, vec![AnsiCommand::StringTerminator]);
}

#[test]
fn it_should_abort_csi_on_esc_and_process_subsequent_csi() {
    let bytes = b"\x1B[1;2\x1B[3m"; // ESC aborts first CSI, second CSI is processed
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Italic
        ]))]
    );
}

#[cfg(test)]
mod unicode_wide_tests {
    use crate::ansi::{AnsiCommand, AnsiParser as AnsiParserTrait, AnsiProcessor}; // Use AnsiParser trait if needed, AnsiProcessor for instantiation
    use std::char; // For char::REPLACEMENT_CHARACTER

    // Import C0Control and EscCommand if they are used in expected AnsiCommand variants
    use crate::ansi::commands::{C0Control, EscCommand};
    use test_log::test;

    // Define byte constants for clarity in tests
    const NUL: u8 = 0x00;
    const ETX: u8 = 0x03;
    const ESC: u8 = 0x1B;
    const BEL_BYTE: u8 = 0x07;
    const C1_PAD_BYTE: u8 = 0x80; // PAD
    const IND_C1_BYTE: u8 = 0x84; // IND
    const ST_C1_BYTE: u8 = 0x9C; // String Terminator

    const CHAR_A_BYTE: u8 = 0x41; // 'A'
    const CHAR_B_BYTE: u8 = 0x42; // 'B'
    const CHAR_C_BYTE: u8 = 0x63; // 'c' (used in RIS)

    // Helper function, assuming AnsiProcessor is the public API to test
    fn process_bytes_unicode(bytes: &[u8]) -> Vec<AnsiCommand> {
        let mut processor = AnsiProcessor::new();
        processor.process_bytes(bytes)
    }

    #[test]
    fn it_should_handle_esc_c_ris_after_interrupted_utf8() {
        let bytes = &[0xE2, ESC, CHAR_C_BYTE]; // Incomplete '€', then ESC c
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Esc(EscCommand::ResetToInitialState), // Assuming from_esc maps 'c' to this
            ],
            "ESC c (RIS) should be processed after UTF-8 interruption"
        );
    }

    #[test]
    fn it_should_handle_c0_bel_and_char_after_interrupted_utf8() {
        let bytes = &[0xF0, BEL_BYTE, CHAR_A_BYTE]; // Incomplete '😀', then BEL, then 'A'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::C0Control(C0Control::BEL),
                AnsiCommand::Print('A'),
            ],
            "BEL and char should be processed after UTF-8 interruption"
        );
    }

    #[test]
    fn it_should_decode_valid_utf8_containing_c1_byte_value_for_ind_and_then_char() {
        let bytes = &[0xE2, 0x82, IND_C1_BYTE, CHAR_B_BYTE]; // E2 82 84 is '₄' (U+2084)
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![AnsiCommand::Print('₄'), AnsiCommand::Print('B'),],
            "0xE2 0x82 0x84 should decode to '₄', not be interrupted by 0x84 as C1 IND"
        );
    }

    #[test]
    fn it_should_handle_complex_interruptions_and_valid_chars_in_sequence() {
        let bytes = &[
            0xE2,        // Start of '€'
            ESC,         // ESC (interrupts '€')
            CHAR_A_BYTE, // 'A' (becomes part of ESC A or Print('A') depending on parser)
            0xF0,
            0x9F,     // Start of '😀'
            BEL_BYTE, // BEL (interrupts '😀')
            0xC2,
            0xA2,        // '¢' (complete char)
            ESC,         // ESC
            CHAR_C_BYTE, // 'c' (RIS)
        ];
        let commands = process_bytes_unicode(bytes);

        let mut expected_commands = vec![
            AnsiCommand::Print(char::REPLACEMENT_CHARACTER), // For interrupted 0xE2
        ];
        // Check how ESC 'A' is handled (assuming it's not a defined sequence, might print 'A')
        if AnsiCommand::from_esc('A').is_none() {
            expected_commands.push(AnsiCommand::Print('A'));
        } else {
            expected_commands.push(AnsiCommand::from_esc('A').unwrap());
        }
        expected_commands.extend(vec![
            AnsiCommand::Print(char::REPLACEMENT_CHARACTER), // For interrupted 0xF0, 0x9F
            AnsiCommand::C0Control(C0Control::BEL),
            AnsiCommand::Print('¢'),
            AnsiCommand::Esc(EscCommand::ResetToInitialState),
        ]);
        assert_eq!(commands, expected_commands);
    }

    #[test]
    fn it_should_process_bel_correctly() {
        let bytes = &[BEL_BYTE];
        let commands = process_bytes_unicode(bytes);
        assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::BEL)]);
    }

    #[test]
    fn it_should_process_esc_c_ris_correctly() {
        let bytes = &[ESC, CHAR_C_BYTE];
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![AnsiCommand::Esc(EscCommand::ResetToInitialState)]
        );
    }

    #[test]
    fn it_should_handle_c0_nul_and_print_after_interrupted_utf8() {
        let bytes = &[0xE2, 0x82, NUL, CHAR_A_BYTE]; // Partial '€', NUL, 'A'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::C0Control(C0Control::NUL),
                AnsiCommand::Print('A'),
            ]
        );
    }

    #[test]
    fn it_should_handle_c0_etx_and_esc_sequence_after_interrupted_utf8() {
        let bytes = &[0xF0, 0x9F, ETX, ESC, b'D']; // Partial '😀', ETX, ESC D (IND)
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::C0Control(C0Control::ETX),
                AnsiCommand::Esc(EscCommand::Index), // Assuming from_esc maps 'D' to Index
            ]
        );
    }

    #[test]
    fn it_should_decode_valid_utf8_containing_c1_byte_value_for_pad_and_then_char() {
        let bytes = &[0xC2, C1_PAD_BYTE, CHAR_A_BYTE]; // C2 80 is U+0080 (PAD control char)
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print('\u{0080}'), // Valid UTF-8 for C1 PAD
                AnsiCommand::Print('A'),
            ]
        );
    }

    #[test]
    fn it_should_handle_c1_st_and_char_after_interrupted_4_byte_utf8() {
        let bytes = &[0xF0, 0x9F, ST_C1_BYTE, CHAR_A_BYTE]; // Partial '😀', C1 ST (0x9C), 'A'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::StringTerminator, // 0x9C is C1 ST
                AnsiCommand::Print('A'),
            ]
        );
    }

    #[test]
    fn it_should_handle_esc_ris_after_3_byte_utf8_interrupted_at_2nd_byte() {
        let bytes = &[0xE2, 0x82, ESC, CHAR_C_BYTE]; // Partial '€', ESC c
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Esc(EscCommand::ResetToInitialState),
            ]
        );
    }

    #[test]
    fn it_should_handle_esc_ris_after_4_byte_utf8_interrupted_at_1st_byte() {
        let bytes = &[0xF0, ESC, CHAR_C_BYTE]; // Partial '😀', ESC c
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Esc(EscCommand::ResetToInitialState),
            ]
        );
    }

    #[test]
    fn it_should_handle_esc_ris_after_4_byte_utf8_interrupted_at_3rd_byte() {
        let bytes = &[0xF0, 0x9F, 0x98, ESC, CHAR_C_BYTE]; // Partial '😀', ESC c
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Esc(EscCommand::ResetToInitialState),
            ]
        );
    }

    #[test]
    fn it_should_handle_double_utf8_interruption_by_esc_then_c0() {
        let bytes = &[0xE2, ESC, b'M', 0xF0, 0x9F, BEL_BYTE]; // Partial '€', ESC M (RI), Partial '😀', BEL
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Esc(EscCommand::ReverseIndex), // Assuming from_esc maps 'M' to RI
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::C0Control(C0Control::BEL),
            ]
        );
    }

    #[test]
    fn it_should_handle_invalid_utf8_continuation_followed_by_chars() {
        let bytes = &[0xE2, 0x41, 0x42]; // 0xE2 (start €), 'A' (invalid cont.), 'B'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER), // For 0xE2 + 0x41 attempt
                AnsiCommand::Print('A'),
                AnsiCommand::Print('B'),
            ]
        );
    }

    #[test]
    fn it_should_handle_overlong_utf8_sequence_c1_af_as_replacement_chars() {
        let bytes = &[0xC1, 0xAF]; // 0xC1 is invalid start, 0xAF is invalid start
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
            ]
        );
    }

    #[test]
    fn it_should_replace_incomplete_3_of_4_byte_utf8_at_stream_end() {
        let bytes = &[0xF0, 0x9F, 0x98]; // Incomplete '😀'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![AnsiCommand::Print(char::REPLACEMENT_CHARACTER),]
        );
    }

    #[test]
    fn it_should_handle_del_interrupting_utf8_then_char() {
        let bytes = &[0xE2, 0x7F, CHAR_A_BYTE]; // Partial '€', DEL (0x7F), 'A'
        let commands = process_bytes_unicode(bytes);
        assert_eq!(
            commands,
            vec![
                AnsiCommand::Print(char::REPLACEMENT_CHARACTER),
                AnsiCommand::C0Control(C0Control::DEL),
                AnsiCommand::Print('A'),
            ]
        );
    }
}
