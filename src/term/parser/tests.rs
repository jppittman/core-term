// src/term/parser/tests.rs

//! Unit tests specifically for the terminal parser state machine logic.
//! These tests focus on state transitions and data collection (params, intermediates, OSC string),
//! not the screen side-effects of executing the sequences.

// Note: Ensure this file is included via `mod tests;` in src/term/parser.rs

#[cfg(test)]
mod parser_tests {
    // Assuming tests are in src/term/parser/tests.rs, relative to src/term/mod.rs:
    use crate::term::{Term, ParserState, MAX_CSI_PARAMS, MAX_OSC_STRING_LEN};
    use crate::glyph::REPLACEMENT_CHARACTER;

    // --- Test Helpers ---

    /// Helper to create a Term instance and process bytes, returning the term.
    fn term_after_bytes(bytes: &[u8]) -> Term {
        let mut term = Term::new(10, 5); // Small dimensions are fine for parser tests
        term.process_bytes(bytes);
        term
    }

    /// Helper to assert final parser state is Ground and buffers are clear.
    fn assert_clean_ground_state(term: &Term, message: &str) {
        assert_eq!(term.parser_state, ParserState::Ground, "{}: Incorrect final state", message);
        // After successfully parsing and handling a sequence, params/intermediates/osc should be cleared.
        assert!(term.csi_params.is_empty(), "{}: CSI params should be cleared", message);
        assert!(term.csi_intermediates.is_empty(), "{}: CSI intermediates should be cleared", message);
        assert!(term.osc_string.is_empty(), "{}: OSC string should be cleared", message);
    }

    // --- Parser State Transition Tests ---

    #[test_log::test]
    fn test_ground_state_printable() {
        let term = term_after_bytes(b"abc");
        assert_clean_ground_state(&term, "Printable chars");
    }

    #[test_log::test]
    fn test_ground_state_c0_control() {
        let term = term_after_bytes(b"\n\r\t\x08"); // LF, CR, TAB, BS
        assert_clean_ground_state(&term, "C0 controls");
    }

    #[test_log::test]
    fn test_ground_to_escape() {
        let term = term_after_bytes(b"\x1b");
        assert_eq!(term.parser_state, ParserState::Escape, "Ground -> Escape");
        assert!(term.csi_params.is_empty(), "Params cleared on ESC entry");
        assert!(term.osc_string.is_empty(), "OSC cleared on ESC entry");
    }

    #[test_log::test]
    fn test_escape_to_csi_entry() {
        let term = term_after_bytes(b"\x1b[");
        assert_eq!(term.parser_state, ParserState::CSIEntry, "Escape -> CSIEntry");
    }

    #[test_log::test]
    fn test_escape_to_osc_string() { // Updated state name
        let term = term_after_bytes(b"\x1b]");
        assert_eq!(term.parser_state, ParserState::OSCString, "Escape -> OSCString"); // Expect OSCString
        assert!(term.osc_string.is_empty(), "OSC string cleared on OSC entry");
    }

    #[test_log::test]
    fn test_escape_single_byte_commands() {
        let term = term_after_bytes(b"\x1bc");
        assert_clean_ground_state(&term, "ESC c (RIS)");
        let term = term_after_bytes(b"\x1b7");
        assert_clean_ground_state(&term, "ESC 7 (DECSC)");
        let term = term_after_bytes(b"\x1b8");
        assert_clean_ground_state(&term, "ESC 8 (DECRC)");
        let term = term_after_bytes(b"\x1bD");
        assert_clean_ground_state(&term, "ESC D (IND)");
        let term = term_after_bytes(b"\x1bE");
        assert_clean_ground_state(&term, "ESC E (NEL)");
        let term = term_after_bytes(b"\x1bH");
        assert_clean_ground_state(&term, "ESC H (HTS)");
        let term = term_after_bytes(b"\x1bM");
        assert_clean_ground_state(&term, "ESC M (RI)");
    }

    #[test_log::test]
    fn test_escape_invalid_sequence() {
        let term = term_after_bytes(b"\x1bA");
        assert_clean_ground_state(&term, "ESC followed by printable");
    }

     #[test_log::test]
    fn test_escape_to_utf8_select() {
        let term = term_after_bytes(b"\x1b%");
        assert_clean_ground_state(&term, "ESC % (UTF-8 select)");
    }

    #[test_log::test]
    fn test_escape_to_alt_charset() {
        let term = term_after_bytes(b"\x1b(");
        assert_eq!(term.parser_state, ParserState::Escape, "ESC ( -> Escape (waiting for char)");
        let term = term_after_bytes(b"\x1b(B");
        assert_clean_ground_state(&term, "ESC ( B (G0 Select)");
    }


    #[test_log::test]
    fn test_csi_entry_param() {
        let term = term_after_bytes(b"\x1b[1");
        assert_eq!(term.parser_state, ParserState::CSIParam, "CSIEntry -> CSIParam");
        assert_eq!(term.csi_params, vec![1]);
        assert!(term.csi_intermediates.is_empty());
    }

    #[test_log::test]
    fn test_csi_entry_private_marker() {
        let term_intermediate = term_after_bytes(b"\x1b[?");
        assert_eq!(term_intermediate.parser_state, ParserState::CSIEntry, "CSIEntry private marker state");
        assert!(term_intermediate.csi_params.is_empty());
        assert_eq!(term_intermediate.csi_intermediates, vec!['?'], "Private marker stored");

        // Test the effect by parsing a full private sequence
        let term_full = term_after_bytes(b"\x1b[?25h"); // Show cursor (private)
        assert_clean_ground_state(&term_full, "Private sequence parsed");
        // Intermediates should be cleared after dispatch
        assert!(term_full.csi_intermediates.is_empty(), "Intermediates cleared after private sequence");
    }

    #[test_log::test]
    fn test_csi_entry_intermediate() {
        let term = term_after_bytes(b"\x1b[!"); // Example intermediate
        assert_eq!(term.parser_state, ParserState::CSIIntermediate, "CSIEntry -> CSIIntermediate");
        assert!(term.csi_params.is_empty());
        assert_eq!(term.csi_intermediates, vec!['!']);
    }

    #[test_log::test]
    fn test_csi_entry_final_byte() {
        let term = term_after_bytes(b"\x1b[A"); // CUU
        assert_clean_ground_state(&term, "CSIEntry -> Ground (no params/intermediates)");
    }

    #[test_log::test]
    fn test_csi_param_digit() {
        let term = term_after_bytes(b"\x1b[12");
        assert_eq!(term.parser_state, ParserState::CSIParam, "CSIParam digit");
        assert_eq!(term.csi_params, vec![12]);
    }

    #[test_log::test]
    fn test_csi_param_separator() {
        let term = term_after_bytes(b"\x1b[1;");
        assert_eq!(term.parser_state, ParserState::CSIParam, "CSIParam separator");
        assert_eq!(term.csi_params, vec![1, 0], "Params after separator");
    }

     #[test_log::test]
    fn test_csi_param_multiple() {
        let term = term_after_bytes(b"\x1b[1;23;4");
        assert_eq!(term.parser_state, ParserState::CSIParam, "CSIParam multiple");
        assert_eq!(term.csi_params, vec![1, 23, 4]);
    }

    #[test_log::test]
    fn test_csi_param_leading_separator() {
        let term = term_after_bytes(b"\x1b[;5B"); // CUD with leading semicolon
        assert_clean_ground_state(&term, "CSI leading semicolon");
    }

    #[test_log::test]
    fn test_csi_param_trailing_separator() {
        let term = term_after_bytes(b"\x1b[5;B"); // CUD with trailing semicolon
        assert_clean_ground_state(&term, "CSI trailing semicolon");
    }

     #[test_log::test]
    fn test_csi_param_max_params() {
        let mut seq = b"\x1b[".to_vec();
        for i in 0..MAX_CSI_PARAMS {
            seq.extend_from_slice(i.to_string().as_bytes());
            seq.push(b';');
        }
        seq.extend_from_slice(b"99A"); // Add one more param + final byte

        let term = term_after_bytes(&seq);
        assert_clean_ground_state(&term, "CSIParam max params");
    }


    #[test_log::test]
    fn test_csi_param_to_intermediate() {
        let term = term_after_bytes(b"\x1b[1;2!"); // Params then intermediate
        assert_eq!(term.parser_state, ParserState::CSIIntermediate, "CSIParam -> CSIIntermediate");
        assert_eq!(term.csi_params, vec![1, 2]);
        assert_eq!(term.csi_intermediates, vec!['!']);
    }

    #[test_log::test]
    fn test_csi_param_final_byte() {
        let term = term_after_bytes(b"\x1b[1;2A"); // CUU with params
        assert_clean_ground_state(&term, "CSIParam -> Ground");
    }

    #[test_log::test]
    fn test_csi_intermediate_byte() {
        let term = term_after_bytes(b"\x1b[1;2!"); // Params, intermediates
        // After processing '>', should still be waiting for final byte
        assert_eq!(term.parser_state, ParserState::CSIIntermediate, "CSIIntermediate byte state");
        assert_eq!(term.csi_params, vec![1, 2], "CSIIntermediate byte params");
        assert_eq!(term.csi_intermediates, vec!['!'], "CSIIntermediate byte intermediates");
    }

     #[test_log::test]
    fn test_csi_intermediate_max_intermediates() {
         // Assuming MAX_CSI_INTERMEDIATES is 2
         let term = term_after_bytes(b"\x1b[1!<>A"); // 3 intermediates -> Ignore state -> Ground
         assert_clean_ground_state(&term, "CSIIntermediate max intermediates");
     }


    #[test_log::test]
    fn test_csi_intermediate_final_byte() {
        let term = term_after_bytes(b"\x1b[1!A"); // Param, intermediate, final
        assert_clean_ground_state(&term, "CSIIntermediate -> Ground");
    }

    #[test_log::test]
    fn test_osc_string_collection() { // Updated test name and state
        let term = term_after_bytes(b"\x1b]0;title");
        assert_eq!(term.parser_state, ParserState::OSCString, "OSCString collection state"); // Expect OSCString
        assert_eq!(term.osc_string, "0;title", "OSC string content");
    }

    #[test_log::test]
    fn test_osc_string_max_len() { // Updated test name and state
         let mut seq = b"\x1b]".to_vec();
         let exact_string = "A".repeat(MAX_OSC_STRING_LEN);
         let extra_string = "EXTRA";
         seq.extend_from_slice(exact_string.as_bytes());
         seq.extend_from_slice(extra_string.as_bytes());

         let mut term = Term::new(10, 5);
         term.process_bytes(&seq);

         assert_eq!(term.parser_state, ParserState::OSCString, "OSCString state after max length"); // Expect OSCString
         assert_eq!(term.osc_string.len(), MAX_OSC_STRING_LEN, "OSC string length truncated");
         assert_eq!(term.osc_string, exact_string, "OSC string content truncated");

         term.process_byte(0x07); // BEL terminator
         assert_clean_ground_state(&term, "OSCString max length final state");
    }

    #[test_log::test]
    fn test_osc_termination_bel() {
        let term = term_after_bytes(b"\x1b]0;title\x07"); // BEL termination
        assert_clean_ground_state(&term, "OSC BEL termination");
    }

     #[test_log::test]
    fn test_osc_termination_st() {
        let term = term_after_bytes(b"\x1b]0;title\x1b\\"); // ST termination (\e\)
        assert_clean_ground_state(&term, "OSC ST termination");
    }

    #[test_log::test]
    fn test_osc_termination_st_alternative() {
        let term = term_after_bytes(b"\x1b]0;title\x9c"); // ST termination (C1)
        assert_clean_ground_state(&term, "OSC ST (C1) termination");
    }

    #[test_log::test]
    fn test_osc_interrupted_by_escape() { // Updated state name
        let mut term = term_after_bytes(b"\x1b]0;tit");
        assert_eq!(term.parser_state, ParserState::OSCString, "OSC state before interrupt"); // Expect OSCString
        term.process_byte(b'\x1b'); // Interrupt with ESC
        assert_eq!(term.parser_state, ParserState::Escape, "OSC interrupted -> Escape");
        assert!(term.osc_string.is_empty(), "OSC string cleared on interrupt");
    }

    #[test_log::test]
    fn test_csi_interrupted_by_escape() {
        let mut term = term_after_bytes(b"\x1b[1;");
        assert_eq!(term.parser_state, ParserState::CSIParam);
        term.process_byte(b'\x1b'); // Interrupt with ESC
        // Expect state to become Escape, and params/intermediates to be cleared
        assert_eq!(term.parser_state, ParserState::Escape, "CSI interrupted -> Escape");
        assert!(term.csi_params.is_empty(), "CSI params cleared on interrupt");
        assert!(term.csi_intermediates.is_empty(), "CSI intermediates cleared on interrupt");
    }

    #[test_log::test]
    fn test_utf8_handling_in_parser() {
        let term = term_after_bytes(b"\x1b]0;titre \xc3\xa9\x07"); // OSC with UTF-8
        assert_clean_ground_state(&term, "OSC with UTF-8");

        let term = term_after_bytes(b"ABC\xc3\xa9FG"); // Ground state with UTF-8
        assert_clean_ground_state(&term, "Ground state with UTF-8");
    }

    #[test_log::test]
    fn test_invalid_utf8_replacement_in_ground() {
         let mut term = Term::new(10, 5);
         term.process_bytes(b"A\x80B"); // Invalid sequence \x80 between A and B
         assert_clean_ground_state(&term, "Invalid UTF-8 in Ground state");
         assert_eq!(term.get_glyph(0, 0).unwrap().c, 'A');
         assert_eq!(term.get_glyph(1, 0).unwrap().c, REPLACEMENT_CHARACTER);
         assert_eq!(term.get_glyph(2, 0).unwrap().c, 'B');
     }

    #[test_log::test]
    fn test_invalid_utf8_within_csi_param() {
         let mut term = Term::new(10, 5);
         // Use 0x80 (C1 control) as the invalid byte within CSI
         term.process_bytes(b"A\x1b[1\x80B"); // Start CSI, param 1, invalid C1, then B
         // Expect parser to reset to Ground upon seeing 0x80, consuming it, then process 'B'.
         assert_clean_ground_state(&term, "Invalid UTF-8 within CSI param");
         assert_eq!(term.get_glyph(0, 0).unwrap().c, 'A'); // A should be printed
         assert_eq!(term.get_glyph(1, 0).unwrap().c, 'B'); // B should be printed at (1,0)
         assert_eq!(term.cursor.x, 2, "Cursor position after B");
     }

    #[test]
    fn test_invalid_utf8_within_osc_string() {
         let mut term = Term::new(10, 5);
         // Use 0x80 (C1 control) as the "invalid" byte within OSC string
         term.process_bytes(b"A\x1b]0;Ti\x80tle\x07B");
         // Expect parser to ignore 0x80 within OSC, finish OSC on BEL, then process 'B'.
         assert_clean_ground_state(&term, "Invalid UTF-8 within OSC string");
         assert_eq!(term.get_glyph(0, 0).unwrap().c, 'A');
         // OSC doesn't print. 'B' prints at cursor pos left after 'A'.
         assert_eq!(term.get_glyph(1, 0).unwrap().c, 'B');
         assert_eq!(term.cursor.x, 2, "Cursor position after B");
     }
}
