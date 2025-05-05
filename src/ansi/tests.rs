// src/ansi/tests.rs

// Import the public AnsiProcessor and command types from the ansi module.
// This file is typically located at src/ansi/tests.rs and tests the public API
// of the `ansi` module declared in src/ansi/mod.rs.
use crate::ansi::{AnsiProcessor, AnsiCommand, CsiCommand, C0Control};

// Helper function to create a processor and process bytes for a single test case
fn process(bytes: &[u8]) -> Vec<AnsiCommand> {
    let mut processor = AnsiProcessor::new();
    processor.process_bytes(bytes)
}

// --- Basic Character and Control Tests ---

#[test]
fn test_process_empty_input() {
    let commands = process(b"");
    assert!(commands.is_empty());
}

#[test]
fn test_process_simple_string() {
    let bytes = b"Hello, world!";
    let commands = process(bytes);
    let expected_commands: Vec<AnsiCommand> = bytes.iter()
        .map(|&b| AnsiCommand::Print(b as char))
        .collect();
    assert_eq!(commands, expected_commands);
}

#[test]
fn test_process_c0_control_bel() {
    let commands = process(b"\x07"); // BEL
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::BEL)]);
}

#[test]
fn test_process_c0_control_bs() {
    let commands = process(b"\x08"); // BS
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::BS)]);
}

#[test]
fn test_process_c0_control_ht() {
    let commands = process(b"\x09"); // HT
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::HT)]);
}

#[test]
fn test_process_c0_control_lf() {
    let commands = process(b"\x0A"); // LF
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::LF)]);
}

#[test]
fn test_process_c0_control_vt() {
    let commands = process(b"\x0B"); // VT
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::VT)]);
}

#[test]
fn test_process_c0_control_ff() {
    let commands = process(b"\x0C"); // FF
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::FF)]);
}

#[test]
fn test_process_c0_control_cr() {
    let commands = process(b"\x0D"); // CR
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::CR)]);
}

#[test]
fn test_process_c0_control_esc() {
    let commands = process(b"\x1B"); // ESC
    // ESC in ground state without a follower is often ignored or an error depending on implementation.
    // Based on the parser logic, an unexpected Escape token results in an Error(0x1B).
    assert_eq!(commands, vec![AnsiCommand::Error(0x1B)]);
}

#[test]
fn test_process_c0_control_del() {
    let commands = process(b"\x7F"); // DEL
    assert_eq!(commands, vec![AnsiCommand::C0Control(C0Control::DEL)]);
}

// --- C1 Control Tests (8-bit and ESC + Fe/Fs) ---

#[test]
fn test_process_c1_ind_8bit() {
    let commands = process(b"\x84"); // IND (8-bit C1)
    // Assuming parser maps 8-bit C1 to CSI commands where applicable
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorNextLine(1))]);
}

#[test]
fn test_process_c1_ind_esc_d() {
    let commands = process(b"\x1BD"); // ESC D (IND)
     // Assuming parser maps ESC D to CSI CursorNextLine(1)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorNextLine(1))]);
}

#[test]
fn test_process_c1_nel_8bit() {
    let commands = process(b"\x85"); // NEL (8-bit C1)
     // Assuming parser maps 8-bit C1 to CSI commands where applicable
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorNextLine(1))]); // NEL often treated like IND
}

#[test]
fn test_process_c1_nel_esc_e() {
    let commands = process(b"\x1BE"); // ESC E (NEL)
     // Assuming parser maps ESC E to CSI CursorNextLine(1)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorNextLine(1))]);
}

#[test]
fn test_process_c1_hts_8bit() {
    let commands = process(b"\x88"); // HTS (8-bit C1)
     // Assuming parser maps 8-bit C1 to CSI commands where applicable
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetTabStop)]);
}

#[test]
fn test_process_c1_hts_esc_h() {
    let commands = process(b"\x1BH"); // ESC H (HTS)
     // Assuming parser maps ESC H to CSI SetTabStop
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetTabStop)]);
}

#[test]
fn test_process_c1_ri_8bit() {
    let commands = process(b"\x8D"); // RI (8-bit C1)
     // Assuming parser maps 8-bit C1 to CSI commands where applicable
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPrevLine(1))]);
}

#[test]
fn test_process_c1_ri_esc_m() {
    let commands = process(b"\x1BM"); // ESC M (RI)
     // Assuming parser maps ESC M to CSI CursorPrevLine(1)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPrevLine(1))]);
}

#[test]
fn test_process_c1_ss2_8bit() {
    let commands = process(b"\x8E"); // SS2 (8-bit C1)
     // Assuming parser maps 8-bit C1 to CSI commands where applicable (SCS G2)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SelectCharacterSet('(', 'N'))]);
}

#[test]
fn test_process_c1_ss2_esc_n() {
    let commands = process(b"\x1BN"); // ESC N (SS2)
     // Assuming parser maps ESC N to CSI SelectCharacterSet('(', 'N')
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SelectCharacterSet('(', 'N'))]);
}

#[test]
fn test_process_c1_ss3_8bit() {
    let commands = process(b"\x8F"); // SS3 (8-bit C1)
     // Assuming parser maps 8-bit C1 to CSI commands where applicable (SCS G3)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SelectCharacterSet(')', 'O'))]);
}

#[test]
fn test_process_c1_ss3_esc_o() {
    let commands = process(b"\x1BO"); // ESC O (SS3)
     // Assuming parser maps ESC O to CSI SelectCharacterSet(')', 'O')
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SelectCharacterSet(')', 'O'))]);
}

#[test]
fn test_process_c1_st_8bit() {
    let commands = process(b"\x9C"); // ST (8-bit C1)
    assert_eq!(commands, vec![AnsiCommand::StringTerminator]);
}

#[test]
fn test_process_c1_st_esc_backslash() {
    let commands = process(b"\x1B\\"); // ESC \ (ST)
    assert_eq!(commands, vec![AnsiCommand::StringTerminator]);
}


// --- CSI (Control Sequence Introducer) Tests ---

#[test]
fn test_process_csi_cuu_no_param() {
    let commands = process(b"\x1B[A"); // CUU (Cursor Up 1 - default)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorUp(1))]);
}

#[test]
fn test_process_csi_cuu_with_param() {
    let commands = process(b"\x1B[5A"); // CUU (Cursor Up 5)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorUp(5))]);
}

#[test]
fn test_process_csi_cud_with_param() {
    let commands = process(b"\x1B[3B"); // CUD (Cursor Down 3)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorDown(3))]);
}

#[test]
fn test_process_csi_cuf_with_param() {
    let commands = process(b"\x1B[10C"); // CUF (Cursor Forward 10)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorForward(10))]);
}

#[test]
fn test_process_csi_cub_with_param() {
    let commands = process(b"\x1B[7D"); // CUB (Cursor Backward 7)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorBackward(7))]);
}

#[test]
fn test_process_csi_cnl_with_param() {
    let commands = process(b"\x1B[2E"); // CNL (Cursor Next Line 2)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorNextLine(2))]);
}

#[test]
fn test_process_csi_cpl_with_param() {
    let commands = process(b"\x1B[4F"); // CPL (Cursor Prev Line 4)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPrevLine(4))]);
}

#[test]
fn test_process_csi_cha_with_param() {
    let commands = process(b"\x1B[5G"); // CHA (Cursor Character Absolute 5)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorCharacterAbsolute(5))]);
}

#[test]
fn test_process_csi_cup_no_params() {
    let commands = process(b"\x1B[H"); // CUP (Cursor Position 1;1 - default)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPosition(1, 1))]);
}

#[test]
fn test_process_csi_cup_single_param() {
    let commands = process(b"\x1B[10;H"); // CUP (Cursor Position 10;1 - default col)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPosition(10, 1))]);
}

#[test]
fn test_process_csi_cup_two_params() {
    let commands = process(b"\x1B[15;20H"); // CUP (Cursor Position 15;20)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPosition(15, 20))]);
}

#[test]
fn test_process_csi_cup_two_params_reverse_order() {
    let commands = process(b"\x1B[;30H"); // CUP (Cursor Position 1;30 - default row)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::CursorPosition(1, 30))]);
}


#[test]
fn test_process_csi_ed_param_0() {
    let commands = process(b"\x1B[0J"); // ED (Erase in Display, from cursor to end - default)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInDisplay(0))]);
}

#[test]
fn test_process_csi_ed_param_1() {
    let commands = process(b"\x1B[1J"); // ED (Erase in Display, from beginning to cursor)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInDisplay(1))]);
}

#[test]
fn test_process_csi_ed_param_2() {
    let commands = process(b"\x1B[2J"); // ED (Erase in Display, entire display)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInDisplay(2))]);
}

#[test]
fn test_process_csi_el_param_0() {
    let commands = process(b"\x1B[0K"); // EL (Erase in Line, from cursor to end - default)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInLine(0))]);
}

#[test]
fn test_process_csi_el_param_1() {
    let commands = process(b"\x1B[1K"); // EL (Erase in Line, from beginning to cursor)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInLine(1))]);
}

#[test]
fn test_process_csi_el_param_2() {
    let commands = process(b"\x1B[2K"); // EL (Erase in Line, entire line)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseInLine(2))]);
}

#[test]
fn test_process_csi_il_no_param() {
    let commands = process(b"\x1B[L"); // IL (Insert Line 1 - default)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::InsertLine(1))]);
}

#[test]
fn test_process_csi_il_with_param() {
    let commands = process(b"\x1B[3L"); // IL (Insert Line 3)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::InsertLine(3))]);
}

#[test]
fn test_process_csi_dl_no_param() {
    let commands = process(b"\x1B[M"); // DL (Delete Line 1 - default)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::DeleteLine(1))]);
}

#[test]
fn test_process_csi_dl_with_param() {
    let commands = process(b"\x1B[2M"); // DL (Delete Line 2)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::DeleteLine(2))]);
}

#[test]
fn test_process_csi_dch_no_param() {
    let commands = process(b"\x1B[P"); // DCH (Delete Character 1 - default)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::DeleteCharacter(1))]);
}

#[test]
fn test_process_csi_dch_with_param() {
    let commands = process(b"\x1B[5P"); // DCH (Delete Character 5)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::DeleteCharacter(5))]);
}

#[test]
fn test_process_csi_ech_no_param() {
    let commands = process(b"\x1B[X"); // ECH (Erase Character 1 - default)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseCharacter(1))]);
}

#[test]
fn test_process_csi_ech_with_param() {
    let commands = process(b"\x1B[4X"); // ECH (Erase Character 4)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::EraseCharacter(4))]);
}

#[test]
fn test_process_csi_ich_no_param() {
    let commands = process(b"\x1B[@"); // ICH (Insert Character 1 - default)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::InsertCharacter(1))]);
}

#[test]
fn test_process_csi_ich_with_param() {
    let commands = process(b"\x1B[6@"); // ICH (Insert Character 6)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::InsertCharacter(6))]);
}


#[test]
fn test_process_csi_sgr_reset() {
    let commands = process(b"\x1B[0m"); // SGR (Reset - default)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![0]))]);
}

#[test]
fn test_process_csi_sgr_foreground() {
    let commands = process(b"\x1B[34m"); // SGR (Set Foreground Blue)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![34]))]);
}

#[test]
fn test_process_csi_sgr_background() {
    let commands = process(b"\x1B[42m"); // SGR (Set Background Green)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![42]))]);
}

#[test]
fn test_process_csi_sgr_attributes() {
    let commands = process(b"\x1B[1;4;7m"); // SGR (Bold, Underline, Inverse)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![1, 4, 7]))]);
}

#[test]
fn test_process_csi_sgr_mixed() {
    let commands = process(b"\x1B[31;1;4m"); // SGR (Foreground Red, Bold, Underline)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![31, 1, 4]))]);
}

#[test]
fn test_process_csi_sgr_truecolor() {
    // SGR True Color: ESC [ 38;2;R;G;Bm or ESC [ 48;2;R;G;Bm
    let commands_fg = process(b"\x1B[38;2;255;100;0m"); // Foreground True Color
    assert_eq!(commands_fg, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![38, 2, 255, 100, 0]))]);

    let commands_bg = process(b"\x1B[48;2;50;200;150m"); // Background True Color
    assert_eq!(commands_bg, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![48, 2, 50, 200, 150]))]);
}

#[test]
fn test_process_csi_sgr_256color() {
    // SGR 256 Color: ESC [ 38;5;n m or ESC [ 48;5;n m
    let commands_fg = process(b"\x1B[38;5;100m"); // Foreground 256 Color
    assert_eq!(commands_fg, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![38, 5, 100]))]);

    let commands_bg = process(b"\x1B[48;5;200m"); // Background 256 Color
    assert_eq!(commands_bg, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![48, 5, 200]))]);
}


#[test]
fn test_process_csi_save_restore_cursor_sco() {
    let commands_save = process(b"\x1B[s"); // SCO Save Cursor
    assert_eq!(commands_save, vec![AnsiCommand::Csi(CsiCommand::SaveCursor)]);

    let commands_restore = process(b"\x1B[u"); // SCO Restore Cursor
    assert_eq!(commands_restore, vec![AnsiCommand::Csi(CsiCommand::RestoreCursor)]);
}

#[test]
fn test_process_csi_save_restore_cursor_ansi() {
    let commands_save = process(b"\x1B7"); // DECSC (ESC 7) - Mapped to CSI SaveCursorAnsi
    assert_eq!(commands_save, vec![AnsiCommand::Csi(CsiCommand::SaveCursorAnsi)]);

    let commands_restore = process(b"\x1B8"); // DECRC (ESC 8) - Mapped to CSI RestoreCursorAnsi
    assert_eq!(commands_restore, vec![AnsiCommand::Csi(CsiCommand::RestoreCursorAnsi)]);
}


#[test]
fn test_process_csi_set_mode() {
    let commands = process(b"\x1B[?25h"); // Set mode (Show cursor - private)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::SetMode(vec![25]))]);
}

#[test]
fn test_process_csi_reset_mode() {
    let commands = process(b"\x1B[?25l"); // Reset mode (Hide cursor - private)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::ResetMode(vec![25]))]);
}

#[test]
fn test_process_csi_dsr() {
    let commands = process(b"\x1B[6n"); // DSR (Report Cursor Position)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::DeviceStatusReport(6))]);
}

#[test]
fn test_process_csi_tbc_param_0() {
    let commands = process(b"\x1B[0g"); // TBC (Clear Tab Stop at current position - default)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::ClearTabStops(0))]);
}

#[test]
fn test_process_csi_tbc_param_3() {
    let commands = process(b"\x1B[3g"); // TBC (Clear All Tab Stops)
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::ClearTabStops(3))]);
}

#[test]
fn test_process_csi_ris() {
    let commands = process(b"\x1Bc"); // RIS (Reset to Initial State) - Mapped to CSI Reset
    assert_eq!(commands, vec![AnsiCommand::Csi(CsiCommand::Reset)]);
}


// --- Fragmented Sequence Tests ---

#[test]
fn test_process_fragmented_csi_split_esc() {
    let mut processor = AnsiProcessor::new();
    let bytes1 = b"\x1B";
    let bytes2 = b"[10;20H";

    let commands1 = processor.process_bytes(bytes1);
    assert!(commands1.is_empty()); // ESC alone doesn't produce a command

    let commands2 = processor.process_bytes(bytes2);
    assert_eq!(commands2, vec![AnsiCommand::Csi(CsiCommand::CursorPosition(10, 20))]);
}

#[test]
fn test_process_fragmented_csi_split_params() {
    let mut processor = AnsiProcessor::new();
    let bytes1 = b"\x1B[31;";
    let bytes2 = b"47m";

    let commands1 = processor.process_bytes(bytes1);
    assert!(commands1.is_empty());

    let commands2 = processor.process_bytes(bytes2);
    assert_eq!(commands2, vec![AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![31, 47]))]);
}

#[test]
fn test_process_fragmented_csi_split_intermediate() {
    let mut processor = AnsiProcessor::new();
    let bytes1 = b"\x1B[?";
    let bytes2 = b"25h";

    let commands1 = processor.process_bytes(bytes1);
    assert!(commands1.is_empty());

    let commands2 = processor.process_bytes(bytes2);
    assert_eq!(commands2, vec![AnsiCommand::Csi(CsiCommand::SetMode(vec![25]))]);
}

#[test]
fn test_process_fragmented_string_with_print() {
    let mut processor = AnsiProcessor::new();
    let bytes1 = b"Hello \x1B[31m"; // String + start of CSI
    let bytes2 = b"world\x1B[0m"; // Rest of CSI + String + Reset SGR

    let commands1 = processor.process_bytes(bytes1);
    let expected1: Vec<AnsiCommand> = b"Hello ".iter().map(|&b| AnsiCommand::Print(b as char)).collect();
    assert_eq!(commands1, expected1); // Only "Hello " should be processed

    let commands2 = processor.process_bytes(bytes2);
    let mut expected2: Vec<AnsiCommand> = b"world".iter().map(|&b| AnsiCommand::Print(b as char)).collect();
    expected2.insert(0, AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![31]))); // SGR from bytes1 + bytes2
    expected2.push(AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(vec![0]))); // Reset SGR
    assert_eq!(commands2, expected2);
}


// --- String (DCS, OSC, PM, APC) Tests ---

#[test]
fn test_process_osc_string() {
    let commands = process(b"\x1B]0;Window Title\x07"); // OSC 0; + string + BEL
    // Assuming parser collects the string bytes
    assert_eq!(commands, vec![AnsiCommand::Osc(b"0;Window Title".to_vec()), AnsiCommand::C0Control(C0Control::BEL)]);
}

#[test]
fn test_process_osc_string_with_st() {
    let commands = process(b"\x1B]0;Window Title\x1B\\"); // OSC 0; + string + ESC \ (ST)
    // Assuming parser collects the string bytes
    assert_eq!(commands, vec![AnsiCommand::Osc(b"0;Window Title".to_vec()), AnsiCommand::StringTerminator]);
}

#[test]
fn test_process_dcs_string() {
    let commands = process(b"\x1BP1;1$rText\x1B\\"); // DCS 1;1$r + string + ST
    // Assuming parser collects the string bytes and parameters/intermediates are handled internally
    // The example parser implementation collects all bytes in DCSPassThrough into dcs_bytes.
    // A more advanced parser would parse the parameters and command within DCS.
    // For this test, we assert the raw bytes are collected.
    assert_eq!(commands, vec![AnsiCommand::Dcs(b"1;1$rText".to_vec()), AnsiCommand::StringTerminator]);
}

#[test]
fn test_process_pm_string() {
    let commands = process(b"\x1B^Privacy Message\x1B\\"); // PM + string + ST
    assert_eq!(commands, vec![AnsiCommand::Pm(b"Privacy Message".to_vec()), AnsiCommand::StringTerminator]);
}

#[test]
fn test_process_apc_string() {
    let commands = process(b"\x1B_Application Command\x1B\\"); // APC + string + ST
    assert_eq!(commands, vec![AnsiCommand::Apc(b"Application Command".to_vec()), AnsiCommand::StringTerminator]);
}


// --- Error and Malformed Sequence Tests ---

#[test]
fn test_process_incomplete_csi() {
    let commands = process(b"\x1B[31"); // Incomplete SGR
    assert!(commands.is_empty()); // Should not produce a command yet
}

#[test]
fn test_process_incomplete_csi_with_more_input() {
    let mut processor = AnsiProcessor::new();
    let bytes1 = b"\x1B[31";
    let bytes2 = b"ABC"; // Non-CSI bytes

    let commands1 = processor.process_bytes(bytes1);
    assert!(commands1.is_empty());

    let commands2 = processor.process_bytes(bytes2);
    // Parser should have aborted the CSI sequence upon seeing 'A' and processed 'A', 'B', 'C' as Print.
    // The partial CSI might be tokenized as Error depending on the parser's error handling.
    // Based on the parser logic, unexpected tokens in CSI state abort the sequence and are tokenized as Error.
    let expected_commands: Vec<AnsiCommand> = vec![
        AnsiCommand::Error(b'A'), // 'A' caused the error and sequence abortion
        AnsiCommand::Print('B'),
        AnsiCommand::Print('C'),
    ];
     // Note: The exact error handling might vary. This test assumes the parser
     // pushes an Error command for the byte that caused the state transition out of CSI.
    assert_eq!(commands2, expected_commands);
}

#[test]
fn test_process_csi_invalid_final_byte() {
    let commands = process(b"\x1B[31a"); // 'a' is not a valid CSI final byte
    // Should be an error or unsupported command
     // Based on the parser logic, unexpected byte in CSI state aborts the sequence and is tokenized as Error.
    assert_eq!(commands, vec![AnsiCommand::Error(b'a')]);
}

#[test]
fn test_process_st_in_ground_state() {
    let commands = process(b"ABC\x9CDEF"); // ST in ground state
    // ST in ground state is typically ignored or an error.
    // Based on the parser logic, it's ignored.
    let mut expected_commands: Vec<AnsiCommand> = b"ABC".iter().map(|&b| AnsiCommand::Print(b as char)).collect();
    expected_commands.push(AnsiCommand::Ignore(0x9C));
    expected_commands.extend(b"DEF".iter().map(|&b| AnsiCommand::Print(b as char)));
    assert_eq!(commands, expected_commands);
}

#[test]
fn test_process_c0_in_csi() {
    let commands = process(b"\x1B[31\x07m"); // BEL in CSI
    // C0 in CSI aborts the sequence and processes the C0.
    let expected_commands = vec![
        AnsiCommand::C0Control(C0Control::BEL),
        AnsiCommand::Print(b'm' as char), // 'm' is processed after CSI is aborted
    ];
    assert_eq!(commands, expected_commands);
}

#[test]
fn test_process_esc_in_csi() {
    let commands = process(b"\x1B[31\x1Bm"); // ESC in CSI
    // ESC in CSI aborts the sequence and processes the ESC (which might then be an error).
     let expected_commands = vec![
        AnsiCommand::C0Control(C0Control::ESC), // ESC token from lexer
        AnsiCommand::Print(b'm' as char), // 'm' is processed after CSI is aborted
    ];
    assert_eq!(commands, expected_commands);
}

#[test]
fn test_process_incomplete_osc() {
    let commands = process(b"\x1B]0;Window Title"); // Missing terminator
    assert!(commands.is_empty()); // Should not produce a command yet
}

#[test]
fn test_process_incomplete_osc_with_more_input() {
    let mut processor = AnsiProcessor::new();
    let bytes1 = b"\x1B]0;Window Title";
    let bytes2 = b"ABC\x07"; // String + BEL terminator

    let commands1 = processor.process_bytes(bytes1);
    assert!(commands1.is_empty());

    let commands2 = processor.process_bytes(bytes2);
    // Should produce the OSC command and the BEL command
    assert_eq!(commands2, vec![AnsiCommand::Osc(b"0;Window TitleABC".to_vec()), AnsiCommand::C0Control(C0Control::BEL)]);
}

#[test]
fn test_process_c0_in_osc() {
    let commands = process(b"\x1B]String\x08with\x07"); // BS in OSC, terminated by BEL
    // C0 controls (except BEL and ESC) are ignored in OSC.
    let expected_commands = vec![
        AnsiCommand::Osc(b"Stringwith".to_vec()), // BS is ignored, string is "Stringwith"
        AnsiCommand::C0Control(C0Control::BEL),
    ];
    assert_eq!(commands, expected_commands);
}

#[test]
fn test_process_esc_in_osc() {
    let commands = process(b"\x1B]String\x1B\x07"); // ESC in OSC, terminated by BEL
    // ESC in OSC aborts the sequence and processes the ESC.
     let expected_commands = vec![
        AnsiCommand::C0Control(C0Control::ESC), // ESC token from lexer
        AnsiCommand::C0Control(C0Control::BEL), // BEL processed after OSC aborted
    ];
    assert_eq!(commands, expected_commands);
}

#[test]
fn test_process_incomplete_dcs() {
    let commands = process(b"\x1BP1;1$rText"); // Missing terminator
    assert!(commands.is_empty()); // Should not produce a command yet
}

#[test]
fn test_process_incomplete_dcs_with_more_input() {
    let mut processor = AnsiProcessor::new();
    let bytes1 = b"\x1BP1;1$rText";
    let bytes2 = b"MoreText\x1B\\"; // String + ST terminator

    let commands1 = processor.process_bytes(bytes1);
    assert!(commands1.is_empty());

    let commands2 = processor.process_bytes(bytes2);
    // Should produce the DCS command and the ST command
    assert_eq!(commands2, vec![AnsiCommand::Dcs(b"1;1$rTextMoreText".to_vec()), AnsiCommand::StringTerminator]);
}

#[test]
fn test_process_c0_in_dcs() {
    let commands = process(b"\x1BPString\x08with\x1B\\"); // BS in DCS, terminated by ST
    // C0 controls (except ESC) are ignored in DCS.
    let expected_commands = vec![
        AnsiCommand::Dcs(b"Stringwith".to_vec()), // BS is ignored, string is "Stringwith"
        AnsiCommand::StringTerminator,
    ];
    assert_eq!(commands, expected_commands);
}

#[test]
fn test_process_esc_in_dcs() {
    let commands = process(b"\x1BPString\x1B\x1B\\"); // ESC in DCS, terminated by ST
    // ESC in DCS aborts the sequence and processes the ESC.
     let expected_commands = vec![
        AnsiCommand::C0Control(C0Control::ESC), // ESC token from lexer
        AnsiCommand::StringTerminator, // ST processed after DCS aborted
    ];
    assert_eq!(commands, expected_commands);
}
