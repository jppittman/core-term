// src/ansi/tests.rs

//! Tests for the ANSI parser and lexer integration.

// Use the AnsiProcessor, which combines lexer and parser
// Corrected imports using commands submodule path
use super::{
    AnsiParser, AnsiProcessor,
    commands::{AnsiCommand, Attribute, C0Control, Color, CsiCommand},
};

// Helper function to process bytes and get commands
fn process_bytes(bytes: &[u8]) -> Vec<AnsiCommand> {
    let mut processor = AnsiProcessor::new();
    processor.process_bytes(bytes)
}

#[test]
fn test_process_simple_string() {
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
fn test_process_c0_control_bel() {
    let bytes = b"\x07"; // BEL
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::BEL)]);
}

// ... (other existing C0, C1, Esc, basic CSI tests remain unchanged) ...

#[test]
fn test_process_csi_cup_no_params() {
    let bytes = b"\x1B[H"; // CSI H -> CUP (1, 1)
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::CursorPosition(1, 1))]
    );
}

#[test]
fn test_process_csi_sgr_reset() {
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
fn test_process_csi_sgr_foreground() {
    let bytes = b"\x1B[34m";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
            Attribute::Foreground(Color::Blue)
        ]))]
    );
}

// --- Tests for sequences from runtime warnings ---

#[test]
fn test_process_dec_private_mode_reset_12_att610_cursor_blink() {
    // CSI ? 12 l - Stop blinking cursor (att610)
    let bytes = b"\x1b[?12l";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(12))],
        "Expected ResetModePrivate(12) for CSI ?12l"
    );
}

#[test]
fn test_process_dec_private_mode_set_25_text_cursor_enable() {
    // CSI ? 25 h - Show cursor (DECTCEM)
    let bytes = b"\x1b[?25h";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(25))],
        "Expected SetModePrivate(25) for CSI ?25h"
    );
}

#[test]
fn test_process_dec_private_mode_reset_25_text_cursor_enable() {
    // CSI ? 25 l - Hide cursor (DECTCEM)
    let bytes = b"\x1b[?25l";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(25))],
        "Expected ResetModePrivate(25) for CSI ?25l"
    );
}

#[test]
fn test_process_dec_private_mouse_modes() {
    let modes_to_test = vec![
        (1000, b"\x1b[?1000h", b"\x1b[?1000l"), // XTERM_MOUSE_CLICK
        (1002, b"\x1b[?1002h", b"\x1b[?1002l"), // XTERM_MOUSE_BTN_MOTION
        (1003, b"\x1b[?1003h", b"\x1b[?1003l"), // XTERM_MOUSE_ANY_MOTION
        (1005, b"\x1b[?1005h", b"\x1b[?1005l"), // XTERM_MOUSE_UTF8
        (1006, b"\x1b[?1006h", b"\x1b[?1006l"), // XTERM_MOUSE_SGR
    ];
    for (mode_num, set_seq, reset_seq) in modes_to_test {
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
fn test_process_dec_private_mode_bracketed_paste_2004() {
    // CSI ? 2004 h - Enable bracketed paste mode
    let bytes_set = b"\x1b[?2004h";
    let commands_set = process_bytes(bytes_set);
    assert_eq!(
        commands_set,
        vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(2004))],
        "Expected SetModePrivate(2004) for CSI ?2004h"
    );

    // CSI ? 2004 l - Disable bracketed paste mode
    let bytes_reset = b"\x1b[?2004l";
    let commands_reset = process_bytes(bytes_reset);
    assert_eq!(
        commands_reset,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(2004))],
        "Expected ResetModePrivate(2004) for CSI ?2004l"
    );
}

#[test]
fn test_process_dec_private_mode_focus_event_1004() {
    // CSI ? 1004 h - Enable focus reporting
    let bytes_set = b"\x1b[?1004h";
    let commands_set = process_bytes(bytes_set);
    assert_eq!(
        commands_set,
        vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(1004))],
        "Expected SetModePrivate(1004) for CSI ?1004h"
    );

    // CSI ? 1004 l - Disable focus reporting
    let bytes_reset = b"\x1b[?1004l";
    let commands_reset = process_bytes(bytes_reset);
    assert_eq!(
        commands_reset,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(1004))],
        "Expected ResetModePrivate(1004) for CSI ?1004l"
    );
}

#[test]
fn test_process_dec_private_mode_uncommon_7727() {
    // CSI ? 7727 l - Example of a less common mode from logs
    let bytes = b"\x1b[?7727l";
    let commands = process_bytes(bytes);
    // This will likely be CsiCommand::Unsupported or an error if not explicitly added to ResetModePrivate
    // For TDD, we expect it to be parsed as ResetModePrivate if we intend to handle it,
    // or as an error/unsupported if we intend the parser to flag it.
    // Let's assume we want to recognize it as a private mode reset.
    assert_eq!(
        commands,
        vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(7727))],
        "Expected ResetModePrivate(7727) for CSI ?7727l"
    );
}

#[test]
fn test_process_csi_set_cursor_style_decscusr() {
    // CSI Ps SP q - DECSCUSR (Set Cursor Style)
    // Example: CSI 2 SP q (Set cursor to steady block)
    // The space is an intermediate character.
    let bytes_steady_block = b"\x1b[2 q";
    let commands_steady_block = process_bytes(bytes_steady_block);
    // This requires a new CsiCommand variant, e.g., SetCursorStyle { shape: u16 }
    // and the parser to handle the space intermediate.
    // For now, this test will fail until that's implemented.
    assert_eq!(
        commands_steady_block,
        vec![AnsiCommand::Csi(CsiCommand::SetCursorStyle { shape: 2 })],
        "Expected SetCursorStyle for CSI 2 SP q"
    );

    // Example: CSI 0 SP q (Reset to default cursor / blinking block by some terms)
    let bytes_default_cursor = b"\x1b[0 q";
    let commands_default_cursor = process_bytes(bytes_default_cursor);
    assert_eq!(
        commands_default_cursor,
        vec![AnsiCommand::Csi(CsiCommand::SetCursorStyle { shape: 0 })],
        "Expected SetCursorStyle for CSI 0 SP q"
    );

    // Example: CSI 3 SP q (Blinking underline)
    let bytes_blink_underline = b"\x1b[3 q";
    let commands_blink_underline = process_bytes(bytes_blink_underline);
    assert_eq!(
        commands_blink_underline,
        vec![AnsiCommand::Csi(CsiCommand::SetCursorStyle { shape: 3 })],
        "Expected SetCursorStyle for CSI 3 SP q"
    );
}

#[test]
fn test_process_csi_window_manipulation_t() {
    // CSI Ps ; Ps ; Ps t - Window manipulation
    // Example from logs: CSI 23 ; 0 ; 0 t
    let bytes_23_0_0_t = b"\x1b[23;0;0t";
    let commands_23_0_0_t = process_bytes(bytes_23_0_0_t);
    // This requires a new CsiCommand variant, e.g., WindowManipulation { ps1: u16, ps2: Option<u16>, ps3: Option<u16> }
    // and the parser to handle it.
    assert_eq!(
        commands_23_0_0_t,
        vec![AnsiCommand::Csi(CsiCommand::WindowManipulation {
            ps1: 23,
            ps2: Some(0),
            ps3: Some(0)
        })],
        "Expected WindowManipulation for CSI 23;0;0t"
    );

    // Example: CSI 18 t (Report text area size in characters)
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

    // Example: CSI 14 t (Report text area size in pixels)
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

// --- Existing Fragmentation and Edge Case Tests remain unchanged ---
// ... (test_process_fragmented_csi_split_params, etc.) ...

#[test]
fn test_process_fragmented_csi_split_params() {
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
fn test_process_fragmented_csi_split_intermediate() {
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
fn test_process_fragmented_csi_split_esc() {
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
fn test_process_fragmented_string_with_print() {
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
                Attribute::Foreground(Color::Red)
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
fn test_process_fragmented_utf8() {
    let mut processor = AnsiProcessor::new();
    let commands_frag1 = processor.process_bytes(b"A");
    assert_eq!(commands_frag1, vec![AnsiCommand::Print('A')], "Frag 0: 'A'");
    let commands_frag2 = processor.process_bytes(b"\xE4");
    assert_eq!(commands_frag2, vec![], "Frag 1: Start of '你' (incomplete)");
    let commands_frag3 = processor.process_bytes(b"\xBD");
    assert_eq!(
        commands_frag3,
        vec![],
        "Frag 2: Middle of '你' (incomplete)"
    );
    let commands_frag4 = processor.process_bytes(b"\xA0");
    assert_eq!(
        commands_frag4,
        vec![AnsiCommand::Print('你')],
        "Frag 3: End of '你' (complete)"
    );
    let commands_frag5 = processor.process_bytes(b"B");
    assert_eq!(commands_frag5, vec![AnsiCommand::Print('B')], "Frag 4: 'B'");
}

#[test]
fn test_process_incomplete_csi_with_more_input() {
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
fn test_process_incomplete_osc_with_more_input() {
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
fn test_process_incomplete_dcs_with_more_input() {
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
fn test_process_osc_string() {
    let bytes = b"\x1B]0;Set Title\x07";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Osc(b"0;Set Title".to_vec())]);
}

#[test]
fn test_process_osc_string_with_st() {
    let bytes = b"\x1B]2;Another Title\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Osc(b"2;Another Title".to_vec())]
    );
}

#[test]
fn test_process_dcs_string() {
    let bytes = b"\x1BP1;1$rText\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Dcs(b"1;1$rText".to_vec())]);
}

#[test]
fn test_process_pm_string() {
    let bytes = b"\x1B^Privacy Message\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Pm(b"Privacy Message".to_vec())]);
}

#[test]
fn test_process_apc_string() {
    let bytes = b"\x1B_Application Command\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Apc(b"Application Command".to_vec())]
    );
}

// --- Edge Case / Error Tests ---

#[test]
fn test_process_empty_input() {
    let bytes = b"";
    let commands = process_bytes(bytes);
    assert!(commands.is_empty());
}

#[test]
fn test_process_incomplete_csi() {
    let bytes = b"\x1B[1;2";
    let commands = process_bytes(bytes);
    assert!(commands.is_empty());
}

#[test]
fn test_process_csi_invalid_final_byte() {
    let bytes = b"\x1B[31a"; // 'a' is not a valid CSI final byte
    let commands = process_bytes(bytes);
    // The parser should ideally produce an Error token or command.
    // If it silently ignores or misinterprets, this test will catch it.
    // Assuming AnsiCommand::Error(final_byte) is the expected outcome.
    assert_eq!(commands, vec![AnsiCommand::Error(b'a')]);
}

#[test]
fn test_process_incomplete_osc() {
    let bytes = b"\x1B]0;Title";
    let commands = process_bytes(bytes);
    assert!(commands.is_empty());
}

#[test]
fn test_process_incomplete_dcs() {
    let bytes = b"\x1BPStuff";
    let commands = process_bytes(bytes);
    assert!(commands.is_empty());
}

#[test]
fn test_process_c0_in_osc() {
    let bytes = b"\x1B]0;String\x08with\x07BEL"; // BEL (0x07) terminates OSC
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::Osc(b"0;String\x08with".to_vec()), // OSC data up to (but not including) BEL
            AnsiCommand::Print('B'), // The 'B' from "BEL" after OSC termination
            AnsiCommand::Print('E'),
            AnsiCommand::Print('L'),
        ]
    );
}

#[test]
fn test_process_esc_in_osc() {
    let bytes = b"\x1B]0;String\x1B\x07BEL"; // ESC (0x1B) aborts OSC
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::C0Control(C0Control::ESC), // The ESC that aborted OSC
            AnsiCommand::C0Control(C0Control::BEL), // BEL processed in Ground state
            AnsiCommand::Print('B'),
            AnsiCommand::Print('E'),
            AnsiCommand::Print('L'),
        ]
    );
}

#[test]
fn test_process_c0_in_dcs() {
    let bytes = b"\x1BPString\x08with\x0BC0\x1B\\"; // C0 controls are part of DCS data
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![AnsiCommand::Dcs(b"String\x08with\x0BC0".to_vec())]
    );
}

#[test]
fn test_process_esc_in_dcs() {
    let bytes = b"\x1BPString\x1B\x1B\\"; // First ESC aborts DCS, second ESC is processed from Ground
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            AnsiCommand::C0Control(C0Control::ESC), // The ESC that aborted DCS
            AnsiCommand::StringTerminator,          // The ESC \ (ST) processed from Ground
        ]
    );
}

#[test]
fn test_process_st_in_ground_state() {
    let bytes = b"\x1B\\"; // ST (ESC \)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::StringTerminator]);

    let bytes_c1 = b"\x9C"; // ST (C1 version)
    let commands_c1 = process_bytes(bytes_c1);
    assert_eq!(commands_c1, vec![AnsiCommand::StringTerminator]);
}

#[test]
fn test_process_esc_in_csi() {
    let bytes = b"\x1B[1;2\x1B[3m"; // ESC aborts first CSI, second CSI is processed
    let commands = process_bytes(bytes);
    assert_eq!(
        commands,
        vec![
            // The first ESC is consumed by the parser to transition from CsiParam/CsiIntermediate to Escape.
            // The second ESC [3m forms a valid CSI sequence.
            AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Italic]))
        ]
    );
}
