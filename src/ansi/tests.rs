// src/ansi/tests.rs

//! Tests for the ANSI parser and lexer integration.

// Use the AnsiProcessor, which combines lexer and parser
// Corrected imports using commands submodule path
use super::{
    AnsiProcessor,
    commands::{AnsiCommand, C0Control, CsiCommand, EscCommand, Attribute, Color}
};


// Helper function to process bytes and get commands
fn process_bytes(bytes: &[u8]) -> Vec<AnsiCommand> {
    let mut processor = AnsiProcessor::new();
    processor.process_bytes(bytes);
    processor.parser.take_commands()
}

// Helper function to process multiple byte slices sequentially
fn process_bytes_fragments(fragments: &[&[u8]]) -> Vec<Vec<AnsiCommand>> {
    let mut processor = AnsiProcessor::new();
    let mut results = Vec::new();
    for frag in fragments {
        processor.process_bytes(frag);
        results.push(processor.parser.take_commands());
    }
    results
}


#[test]
fn test_process_simple_string() {
    let bytes = b"Hello, world!";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![
        AnsiCommand::Print('H'), AnsiCommand::Print('e'), AnsiCommand::Print('l'), AnsiCommand::Print('l'), AnsiCommand::Print('o'),
        AnsiCommand::Print(','), AnsiCommand::Print(' '),
        AnsiCommand::Print('w'), AnsiCommand::Print('o'), AnsiCommand::Print('r'), AnsiCommand::Print('l'), AnsiCommand::Print('d'),
        AnsiCommand::Print('!'),
    ]);
}

#[test]
fn test_process_c0_control_bel() {
    let bytes = b"\x07"; // BEL
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::BEL)]);
}

#[test]
fn test_process_c0_control_lf() {
    let bytes = b"\x0A"; // LF
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::LF)]);
}

#[test]
fn test_process_c0_control_cr() {
    let bytes = b"\x0D"; // CR
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::CR)]);
}

#[test]
fn test_process_c0_control_bs() {
    let bytes = b"\x08"; // BS
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::BS)]);
}

#[test]
fn test_process_c0_control_ht() {
    let bytes = b"\x09"; // HT
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::HT)]);
}

#[test]
fn test_process_c0_control_vt() {
     let bytes = b"\x0B"; // VT
     let commands = process_bytes(bytes);
     assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::VT)]);
}

#[test]
fn test_process_c0_control_ff() {
     let bytes = b"\x0C"; // FF
     let commands = process_bytes(bytes);
     assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::FF)]);
}

#[test]
fn test_process_c0_control_esc() {
    let bytes = b"\x1B"; // ESC
    let commands = process_bytes(bytes);
    assert!(commands.is_empty()); // Parser enters Escape state, no command emitted yet
}

#[test]
fn test_process_c0_control_del() {
    let bytes = b"\x7F"; // DEL
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::DEL)]);
}


// --- C1 Control Tests ---

#[test]
fn test_process_c1_ind_8bit() {
    let bytes = b"\x84"; // IND
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::Index)]);
}

#[test]
fn test_process_c1_ind_esc_d() {
    let bytes = b"\x1BD"; // ESC D
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::Index)]);
}

#[test]
fn test_process_c1_nel_8bit() {
    let bytes = b"\x85"; // NEL
    let commands = process_bytes(bytes);
     assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::NextLine)]);
}

#[test]
fn test_process_c1_nel_esc_e() {
    let bytes = b"\x1BE"; // ESC E
    let commands = process_bytes(bytes);
     assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::NextLine)]);
}

#[test]
fn test_process_c1_hts_8bit() {
    let bytes = b"\x88"; // HTS
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::SetTabStop)]);
}

#[test]
fn test_process_c1_hts_esc_h() {
    let bytes = b"\x1BH"; // ESC H
    let commands = process_bytes(bytes);
     assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::SetTabStop)]);
}

#[test]
fn test_process_c1_ri_8bit() {
    let bytes = b"\x8D"; // RI
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::ReverseIndex)]);
}

#[test]
fn test_process_c1_ri_esc_m() {
    let bytes = b"\x1BM"; // ESC M
    let commands = process_bytes(bytes);
     assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::ReverseIndex)]);
}

#[test]
fn test_process_c1_ss2_8bit() {
    let bytes = b"\x8E"; // SS2
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::SingleShift2)]);
}

#[test]
fn test_process_c1_ss2_esc_n() {
    let bytes = b"\x1BN"; // ESC N
    let commands = process_bytes(bytes);
     assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::SingleShift2)]);
}

#[test]
fn test_process_c1_ss3_8bit() {
    let bytes = b"\x8F"; // SS3
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::SingleShift3)]);
}

#[test]
fn test_process_c1_ss3_esc_o() {
    let bytes = b"\x1BO"; // ESC O
    let commands = process_bytes(bytes);
     assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::SingleShift3)]);
}

#[test]
fn test_process_c1_st_8bit() {
    let bytes = b"\x9C"; // ST
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::StringTerminator]);
}

#[test]
fn test_process_c1_st_esc_backslash() {
    let bytes = b"\x1B\\"; // ESC \
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::StringTerminator]);
}

// --- Select Character Set Tests (Corrected to ESC) ---

#[test]
fn test_process_scs_g0_esc() {
    let bytes = b"\x1B(B"; // ESC ( B -> Select G0 Charset ASCII
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::SelectCharacterSet('(', 'B'))]);
}

#[test]
fn test_process_scs_g1_esc() {
    let bytes = b"\x1B)0"; // ESC ) 0 -> Select G1 Charset DEC Special Graphics
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::SelectCharacterSet(')', '0'))]);
}


// --- CSI Basic Tests ---

#[test]
fn test_process_csi_cup_no_params() {
    let bytes = b"\x1B[H"; // CSI H -> CUP (1, 1)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPosition(1, 1))]);
}

#[test]
fn test_process_csi_cud_with_param() {
    let bytes = b"\x1B[3B"; // CSI 3 B -> CursorDown(3)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorDown(3))]);
}

#[test]
fn test_process_csi_cuf_with_param() {
    let bytes = b"\x1B[10C"; // CSI 10 C -> CursorForward(10)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorForward(10))]);
}

#[test]
fn test_process_csi_cub_with_param() {
    let bytes = b"\x1B[7D"; // CSI 7 D -> CursorBackward(7)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorBackward(7))]);
}

#[test]
fn test_process_csi_cnl_with_param() {
    let bytes = b"\x1B[2E"; // CSI 2 E -> CursorNextLine(2)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorNextLine(2))]);
}

#[test]
fn test_process_csi_cpl_with_param() {
    let bytes = b"\x1B[4F"; // CSI 4 F -> CursorPrevLine(4)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPrevLine(4))]);
}

#[test]
fn test_process_csi_cha_with_param() {
    let bytes = b"\x1B[5G"; // CSI 5 G -> CursorCharacterAbsolute(5)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorCharacterAbsolute(5))]);
}

#[test]
fn test_process_csi_cup_single_param() {
    let bytes = b"\x1B[10H"; // CSI 10 H -> CUP (10, 1)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPosition(10, 1))]);
}

#[test]
fn test_process_csi_cup_two_params() {
    let bytes = b"\x1B[15;20H"; // CSI 15 ; 20 H -> CUP (15, 20)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPosition(15, 20))]);
}

#[test]
fn test_process_csi_cup_two_params_reverse_order() {
     let bytes = b"\x1B[30;1f"; // CSI 30 ; 1 f -> CUP (30, 1) or HVP(30, 1)
     let commands = process_bytes(bytes);
     // Corrected assertion: Both H and f use row;col
     assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPosition(30, 1))]);
}

#[test]
fn test_process_csi_ed_param_0() {
    let bytes = b"\x1B[0J"; // CSI 0 J -> EraseInDisplay(0)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInDisplay(0))]);
}

#[test]
fn test_process_csi_ed_param_1() {
    let bytes = b"\x1B[1J"; // CSI 1 J -> EraseInDisplay(1)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInDisplay(1))]);
}

#[test]
fn test_process_csi_ed_param_2() {
    let bytes = b"\x1B[2J"; // CSI 2 J -> EraseInDisplay(2)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInDisplay(2))]);
}

#[test]
fn test_process_csi_el_param_0() {
    let bytes = b"\x1B[0K"; // CSI 0 K -> EraseInLine(0)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInLine(0))]);
}

#[test]
fn test_process_csi_el_param_1() {
    let bytes = b"\x1B[1K"; // CSI 1 K -> EraseInLine(1)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInLine(1))]);
}

#[test]
fn test_process_csi_el_param_2() {
    let bytes = b"\x1B[2K"; // CSI 2 K -> EraseInLine(2)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInLine(2))]);
}

#[test]
fn test_process_csi_il_no_param() {
    let bytes = b"\x1B[L"; // CSI L -> InsertLine(1)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::InsertLine(1))]);
}

#[test]
fn test_process_csi_il_with_param() {
    let bytes = b"\x1B[3L"; // CSI 3 L -> InsertLine(3)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::InsertLine(3))]);
}

#[test]
fn test_process_csi_dl_no_param() {
    let bytes = b"\x1B[M"; // CSI M -> DeleteLine(1)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::DeleteLine(1))]);
}

#[test]
fn test_process_csi_dl_with_param() {
    let bytes = b"\x1B[2M"; // CSI 2 M -> DeleteLine(2)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::DeleteLine(2))]);
}

#[test]
fn test_process_csi_dch_no_param() {
    let bytes = b"\x1B[P"; // CSI P -> DeleteCharacter(1)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::DeleteCharacter(1))]);
}

#[test]
fn test_process_csi_dch_with_param() {
    let bytes = b"\x1B[5P"; // CSI 5 P -> DeleteCharacter(5)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::DeleteCharacter(5))]);
}

#[test]
fn test_process_csi_ech_no_param() {
    let bytes = b"\x1B[X"; // CSI X -> EraseCharacter(1)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseCharacter(1))]);
}

#[test]
fn test_process_csi_ech_with_param() {
    let bytes = b"\x1B[4X"; // CSI 4 X -> EraseCharacter(4)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseCharacter(4))]);
}

#[test]
fn test_process_csi_ich_no_param() {
    let bytes = b"\x1B[@"; // CSI @ -> InsertCharacter(1)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::InsertCharacter(1))]);
}

#[test]
fn test_process_csi_ich_with_param() {
    let bytes = b"\x1B[6@"; // CSI 6 @ -> InsertCharacter(6)
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::InsertCharacter(6))]);
}

#[test]
fn test_process_csi_sgr_reset() {
    let bytes = b"\x1B[0m";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Reset]))]);
}

#[test]
fn test_process_csi_sgr_foreground() {
    let bytes = b"\x1B[34m";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Blue)]))]);
}

#[test]
fn test_process_csi_sgr_background() {
    let bytes = b"\x1B[42m";
    let commands = process_bytes(bytes);
     assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Background(Color::Green)]))]);
}

#[test]
fn test_process_csi_sgr_attributes() {
    let bytes = b"\x1B[1;4;7m";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
        Attribute::Bold, Attribute::Underline, Attribute::Reverse
    ]))]);
}

#[test]
fn test_process_csi_sgr_mixed() {
    let bytes = b"\x1B[31;1;4m";
    let commands = process_bytes(bytes);
     assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
         Attribute::Foreground(Color::Red), Attribute::Bold, Attribute::Underline
     ]))]);
}

#[test]
fn test_process_csi_sgr_truecolor() {
    let bytes_fg = b"\x1B[38;2;255;100;0m";
    let commands_fg = process_bytes(bytes_fg);
    assert_eq!(commands_fg, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
        Attribute::Foreground(Color::Rgb(255, 100, 0))
    ]))]);

    let bytes_bg = b"\x1B[48;2;50;200;150m";
    let commands_bg = process_bytes(bytes_bg);
     assert_eq!(commands_bg, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
         Attribute::Background(Color::Rgb(50, 200, 150))
     ]))]);
}

#[test]
fn test_process_csi_sgr_256color() {
    let bytes_fg = b"\x1B[38;5;100m";
    let commands_fg = process_bytes(bytes_fg);
     assert_eq!(commands_fg, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
         Attribute::Foreground(Color::Indexed(100))
     ]))]);

    let bytes_bg = b"\x1B[48;5;200m";
    let commands_bg = process_bytes(bytes_bg);
     assert_eq!(commands_bg, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![
         Attribute::Background(Color::Indexed(200))
     ]))]);
}

#[test]
fn test_process_csi_save_restore_cursor_sco() {
    let bytes_save = b"\x1B[s";
    let commands_save = process_bytes(bytes_save);
    assert_eq!(commands_save, vec![AnsiCommand::Csi(CsiCommand::SaveCursor)]);

    let bytes_restore = b"\x1B[u";
    let commands_restore = process_bytes(bytes_restore);
    assert_eq!(commands_restore, vec![AnsiCommand::Csi(CsiCommand::RestoreCursor)]);
}

#[test]
fn test_process_csi_save_restore_cursor_ansi() {
    let bytes_save = b"\x1B7";
    let commands_save = process_bytes(bytes_save);
    assert_eq!(commands_save, vec![AnsiCommand::Esc(EscCommand::SaveCursor)]);

    let bytes_restore = b"\x1B8";
    let commands_restore = process_bytes(bytes_restore);
    assert_eq!(commands_restore, vec![AnsiCommand::Esc(EscCommand::RestoreCursor)]);
}


#[test]
fn test_process_csi_set_mode() {
    let bytes = b"\x1B[?25h";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(25))]);
}

#[test]
fn test_process_csi_reset_mode() {
    let bytes = b"\x1B[?25l";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::ResetModePrivate(25))]);
}

#[test]
fn test_process_csi_dsr() {
    let bytes = b"\x1B[6n";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::DeviceStatusReport(6))]);
}

#[test]
fn test_process_csi_tbc_param_0() {
    let bytes = b"\x1B[0g";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::ClearTabStops(0))]);
}

#[test]
fn test_process_csi_tbc_param_3() {
    let bytes = b"\x1B[3g";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::ClearTabStops(3))]);
}

#[test]
fn test_process_csi_ris() {
    let bytes = b"\x1Bc";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Esc(EscCommand::ResetToInitialState)]);
}


// --- Fragmentation Tests ---

#[test]
fn test_process_fragmented_csi_split_params() {
    let fragments = [b"\x1B[1".as_slice(), b";2H".as_slice()];
    let results = process_bytes_fragments(&fragments);
    assert_eq!(results[0], vec![]);
    assert_eq!(results[1], vec![AnsiCommand::Csi(CsiCommand::CursorPosition(1, 2))]);
}

#[test]
fn test_process_fragmented_csi_split_intermediate() {
     let fragments = [b"\x1B[?".as_slice(), b"25h".as_slice()];
     let results = process_bytes_fragments(&fragments);
     assert_eq!(results[0], vec![]);
     // Should now correctly parse SetModePrivate
     assert_eq!(results[1], vec![AnsiCommand::Csi(CsiCommand::SetModePrivate(25))]);
}

#[test]
fn test_process_fragmented_csi_split_esc() {
     let fragments = [b"\x1B".as_slice(), b"[1A".as_slice()];
     let results = process_bytes_fragments(&fragments);
     assert_eq!(results[0], vec![]);
     assert_eq!(results[1], vec![AnsiCommand::Csi(CsiCommand::CursorUp(1))]);
}


#[test]
fn test_process_fragmented_string_with_print() {
    let fragments = [
        b"Hello ".as_slice(),
        b"\x1B[31".as_slice(),
        b"m World".as_slice(),
    ];
    let results = process_bytes_fragments(&fragments);

    let expected0 = vec![
        AnsiCommand::Print('H'), AnsiCommand::Print('e'), AnsiCommand::Print('l'),
        AnsiCommand::Print('l'), AnsiCommand::Print('o'), AnsiCommand::Print(' '),
    ];
    let expected1 = vec![];
    let expected2 = vec![
        AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Foreground(Color::Red)])),
        AnsiCommand::Print(' '), AnsiCommand::Print('W'), AnsiCommand::Print('o'),
        AnsiCommand::Print('r'), AnsiCommand::Print('l'), AnsiCommand::Print('d'),
    ];

    assert_eq!(results[0], expected0);
    assert_eq!(results[1], expected1);
    assert_eq!(results[2], expected2);
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
     assert_eq!(commands, vec![AnsiCommand::Osc(b"2;Another Title".to_vec())]);
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
     assert_eq!(commands, vec![AnsiCommand::Apc(b"Application Command".to_vec())]);
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
fn test_process_incomplete_csi_with_more_input() {
    let fragments = [b"\x1B[31".as_slice(), b"A".as_slice(), b"BC".as_slice()];
    let results = process_bytes_fragments(&fragments);
    assert_eq!(results[0], vec![]);
    assert_eq!(results[1], vec![AnsiCommand::Csi(CsiCommand::CursorUp(31))]);
    assert_eq!(results[2], vec![AnsiCommand::Print('B'), AnsiCommand::Print('C')]);
}


#[test]
fn test_process_csi_invalid_final_byte() {
    let bytes = b"\x1B[31a";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Error(b'a')]);
}

#[test]
fn test_process_incomplete_osc() {
    let bytes = b"\x1B]0;Title";
    let commands = process_bytes(bytes);
    assert!(commands.is_empty());
}

#[test]
fn test_process_incomplete_osc_with_more_input() {
     let fragments = [b"\x1B]0;Ti".as_slice(), b"tle\x07".as_slice()];
     let results = process_bytes_fragments(&fragments);
     assert_eq!(results[0], vec![]);
     assert_eq!(results[1], vec![AnsiCommand::Osc(b"0;Title".to_vec())]);
}

#[test]
fn test_process_incomplete_dcs() {
    let bytes = b"\x1BPStuff";
    let commands = process_bytes(bytes);
    assert!(commands.is_empty());
}

#[test]
fn test_process_incomplete_dcs_with_more_input() {
     let fragments = [b"\x1BPSt".as_slice(), b"uff\x1B\\".as_slice()];
     let results = process_bytes_fragments(&fragments);
     assert_eq!(results[0], vec![]);
     assert_eq!(results[1], vec![AnsiCommand::Dcs(b"Stuff".to_vec())]);
}


#[test]
fn test_process_c0_in_osc() {
    let bytes = b"\x1B]0;String\x08with\x07BEL";
    let commands = process_bytes(bytes);
    // Corrected assertion: BEL terminates OSC, parser returns to ground, then processes 'B','E','L'
    assert_eq!(commands, vec![
        AnsiCommand::Osc(b"0;String\x08with".to_vec()),
        AnsiCommand::Print('B'), AnsiCommand::Print('E'), AnsiCommand::Print('L'),
    ]);
}

#[test]
fn test_process_esc_in_osc() {
    let bytes = b"\x1B]0;String\x1B\x07BEL";
    let commands = process_bytes(bytes);
    // Corrected assertion: ESC aborts OSC, emits C0(ESC), returns to ground, processes BEL, then 'B','E','L'
    assert_eq!(commands, vec![
        AnsiCommand::C0Control(C0Control::ESC),
        AnsiCommand::C0Control(C0Control::BEL),
        AnsiCommand::Print('B'), AnsiCommand::Print('E'), AnsiCommand::Print('L'),
    ]);
}


#[test]
fn test_process_c0_in_dcs() {
    let bytes = b"\x1BPString\x08with\x0BC0\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::Dcs(b"String\x08with\x0BC0".to_vec())]);
}

#[test]
fn test_process_esc_in_dcs() {
    let bytes = b"\x1BPString\x1B\x1B\\";
    let commands = process_bytes(bytes);
     assert_eq!(commands, vec![
         AnsiCommand::C0Control(C0Control::ESC),
         AnsiCommand::StringTerminator,
     ]);
}

#[test]
fn test_process_st_in_ground_state() {
    let bytes = b"\x1B\\";
    let commands = process_bytes(bytes);
    assert_eq!(commands, vec![AnsiCommand::StringTerminator]);
    let bytes_c1 = b"\x9C";
    let commands_c1 = process_bytes(bytes_c1);
    assert_eq!(commands_c1, vec![AnsiCommand::StringTerminator]);
}

#[test]
fn test_process_esc_in_csi() {
    let bytes = b"\x1B[1;2\x1B[3m";
    let commands = process_bytes(bytes);
    // Corrected assertion: Expect SGR Italic (3m) after ESC aborts the first CSI
    assert_eq!(commands, vec![
        AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![Attribute::Italic]))
    ]);
}
