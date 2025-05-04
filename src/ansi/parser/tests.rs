// src/ansi/parser/tests.rs

//! Unit tests for the ANSI parser state machine and command generation.
//!
//! These tests focus *solely* on verifying that the AnsiParser correctly
//! consumes byte sequences via its public `feed` and `feed_slice` methods
//! and produces the expected sequence of AnsiCommand enums as output.
//! They do not test internal parser state transitions or buffer contents,
//! adhering to the style guide's recommendation to test public APIs.

#[cfg(test)]
mod parser_command_tests {
    // Use super to access items from the parent module (parser.rs)
    use super::super::AnsiParser;
    // Use items from the commands module
    use crate::ansi::commands::{AnsiCommand, C0Control, CsiCommand, OscCommand, SgrParameter, EraseMode, DecPrivateMode};
    // Use log for test output (optional, but good for debugging failed tests)
    use log::LevelFilter;

    // Helper to initialize logging for tests
    fn init_logging() {
        // Using try_init() to avoid panicking if logging is already initialized
        let _ = env_logger::builder()
            .filter_level(LevelFilter::Trace) // Set to Trace for detailed parser logs
            .is_test(true)
            .try_init();
    }

    // Helper to create a parser, feed bytes, and return the generated commands.
    // This is the primary way to interact with the parser's public API in tests.
    fn parse_bytes(bytes: &[u8]) -> Vec<AnsiCommand> {
        init_logging(); // Initialize logging for each test
        let mut parser = AnsiParser::new();
        parser.feed_slice(bytes)
    }

    // Helper to create a parser, feed bytes one by one, and return the generated commands.
    // Useful for testing incremental input and buffering.
    fn parse_bytes_incremental(bytes: &[u8]) -> Vec<AnsiCommand> {
        init_logging();
        let mut parser = AnsiParser::new();
        let mut all_commands = Vec::new();
        for &byte in bytes {
            all_commands.extend(parser.feed(byte));
        }
        all_commands
    }


    // --- Basic Tests ---

    #[test]
    fn test_printable_chars() {
        let commands = parse_bytes(b"Hello, world!");
        assert_eq!(commands.len(), 13);
        assert_eq!(commands, vec![
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
        ]);
    }

    #[test]
    fn test_c0_controls() {
        let commands = parse_bytes(&[0x07, 0x08, 0x09, 0x0A, 0x0D, 0x1A, 0x7F]); // BEL, BS, HT, LF, CR, SUB, DEL
        assert_eq!(commands.len(), 7);
        assert_eq!(commands, vec![
            AnsiCommand::C0Control(C0Control::Bell),
            AnsiCommand::C0Control(C0Control::Backspace),
            AnsiCommand::C0Control(C0Control::Tab),
            AnsiCommand::C0Control(C0Control::LineFeed),
            AnsiCommand::C0Control(C0Control::CarriageReturn),
            AnsiCommand::C0Control(C0Control::Substitute),
            AnsiCommand::C0Control(C0Control::Delete),
        ]);
    }

     #[test]
    fn test_c1_controls() {
        // Test C1 controls that map to CSI or specific commands
        let commands = parse_bytes(&[0x84, 0x85, 0x88, 0x8D]); // IND, NEL, HTS, RI
        assert_eq!(commands.len(), 4);
        assert_eq!(commands, vec![
            AnsiCommand::Csi(CsiCommand::CursorNextLine(1)), // C1 IND (0x84) -> CSI 1 E (approx)
            AnsiCommand::Csi(CsiCommand::CursorNextLine(1)), // C1 NEL (0x85) -> CSI 1 E (approx)
            AnsiCommand::Csi(CsiCommand::SetTabStop), // C1 HTS (0x88) -> CSI H
            AnsiCommand::Csi(CsiCommand::CursorPrevLine(1)), // C1 RI (0x8D) -> CSI 1 F (approx)
        ]);

         // Test C1 controls that start sequences (CSI, OSC)
         let commands_csi_c1 = parse_bytes(&[0x9b, b'2', b'J']); // C1 CSI + 2J
         assert_eq!(commands_csi_c1.len(), 1);
         assert_eq!(commands_csi_c1, vec![AnsiCommand::Csi(CsiCommand::EraseDisplay(EraseMode::All))]);

         let commands_osc_c1 = parse_bytes(&[0x9d, b'0', b';', b't', b'i', b't', b'l', b'e', 0x9c]); // C1 OSC + 0;title + C1 ST
         assert_eq!(commands_osc_c1.len(), 1);
         assert_eq!(commands_osc_c1, vec![AnsiCommand::Osc(OscCommand::SetIconAndWindowTitle("0;title".to_string()))]);

         // Test C1 controls that are currently ignored/unknown
         let commands_ignored_c1 = parse_bytes(&[0x80, 0x81, 0x82, 0x9e, 0x9f]);
          assert_eq!(commands_ignored_c1.len(), 5);
          assert_eq!(commands_ignored_c1, vec![
             AnsiCommand::Ignore(0x80),
             AnsiCommand::Ignore(0x81),
             AnsiCommand::Ignore(0x82),
             AnsiCommand::Ignore(0x9e),
             AnsiCommand::Ignore(0x9f),
         ]);
     }


    #[test]
    fn test_escape_single_byte_sequences() {
        let commands = parse_bytes(b"\x1b7\x1b8\x1bD\x1bE\x1bH\x1bM\x1bc"); // DECSC, DECRC, IND, NEL, HTS, RI, RIS
        assert_eq!(commands.len(), 6); // RIS (ESC c) currently not mapped to a command
        assert_eq!(commands, vec![
            AnsiCommand::Csi(CsiCommand::SaveCursor),         // ESC 7 (DECSC)
            AnsiCommand::Csi(CsiCommand::RestoreCursor),       // ESC 8 (DECRC)
            AnsiCommand::Csi(CsiCommand::CursorNextLine(1)),   // ESC D (IND)
            AnsiCommand::Csi(CsiCommand::CursorNextLine(1)),   // ESC E (NEL)
            AnsiCommand::Csi(CsiCommand::SetTabStop),          // ESC H (HTS)
            AnsiCommand::Csi(CsiCommand::CursorPrevLine(1)),   // ESC M (RI)
            // AnsiCommand::EscapeSequence('c', vec![]), // RIS (ESC c) - if implemented as generic
        ]);
    }

    #[test]
    fn test_escape_charset_designation_sequences() {
        // ESC ( G0 Designation
        let commands_g0 = parse_bytes(b"\x1b(B"); // ESC ( B -> Select G0 as US-ASCII
        assert!(commands_g0.is_empty(), "Charset designation commands are not yet implemented");

        // ESC ) G1 Designation
        let commands_g1 = parse_bytes(b"\x1b)0"); // ESC ) 0 -> Select G1 as DEC Special Graphics
        assert!(commands_g1.is_empty(), "Charset designation commands are not yet implemented");

         // Test incomplete sequences
         let commands_incomplete = parse_bytes_incremental(b"\x1b("); // ESC (
         assert!(commands_incomplete.is_empty());
         let commands_complete = parse_bytes_incremental(b"B"); // B
         assert!(commands_complete.is_empty(), "Charset commands not emitted yet");
     }

    // --- CSI Sequence Tests (Parameters, Intermediates, Final Bytes) ---

    #[test]
    fn test_csi_cursor_movement_commands() {
        let commands = parse_bytes(b"\x1b[5A\x1b[10B\x1b[3C\x1b[2D\x1b[1E\x1b[2F\x1b[1G\x1b[10;20H\x1b[5d");
        assert_eq!(commands.len(), 9);
        assert_eq!(commands, vec![
            AnsiCommand::Csi(CsiCommand::CursorUp(5)),
            AnsiCommand::Csi(CsiCommand::CursorDown(10)),
            AnsiCommand::Csi(CsiCommand::CursorForward(3)),
            AnsiCommand::Csi(CsiCommand::CursorBackward(2)),
            AnsiCommand::Csi(CsiCommand::CursorNextLine(1)),
            AnsiCommand::Csi(CsiCommand::CursorPrevLine(2)),
            AnsiCommand::Csi(CsiCommand::CursorHorizontalAbsolute(1)),
            AnsiCommand::Csi(CsiCommand::CursorPosition { row: 10, col: 20 }),
            AnsiCommand::Csi(CsiCommand::VerticalLineAbsolute(5)),
        ]);
    }

    #[test]
    fn test_csi_parameters() {
        // Test various parameter formats including defaults
        let commands_defaults = parse_bytes(b"\x1b[A\x1b[;B\x1b[C\x1b[D\x1b[H"); // CUU, CUD, CUF, CUB, CUP with default params
        assert_eq!(commands_defaults.len(), 5);
        assert_eq!(commands_defaults, vec![
            AnsiCommand::Csi(CsiCommand::CursorUp(1)), // Default param is 1
            AnsiCommand::Csi(CsiCommand::CursorDown(1)), // Leading ; implies default 1
            AnsiCommand::Csi(CsiCommand::CursorForward(1)), // No param implies default 1
            AnsiCommand::Csi(CsiCommand::CursorBackward(1)), // No param implies default 1
            AnsiCommand::Csi(CsiCommand::CursorPosition { row: 1, col: 1 }), // No params implies default 1,1
        ]);

         // Test multiple parameters and trailing/leading semicolons on explicit params
         let commands_params = parse_bytes(b"\x1b[1;2;3B\x1b[5;C\x1b[;10D");
         assert_eq!(commands_params.len(), 3);
         assert_eq!(commands_params, vec![
             AnsiCommand::Csi(CsiCommand::CursorDown(3)), // CUD takes only the first param
             AnsiCommand::Csi(CsiCommand::CursorForward(5)), // Trailing ; doesn't affect param count/value
             AnsiCommand::Csi(CsiCommand::CursorBackward(10)), // Leading ; gives 0, then 10, but CUB takes first param
         ]);

         // Test maximum parameters (should likely truncate and still dispatch)
         let max_params = 16; // From ansi/mod.rs
         let mut seq_max_params = b"\x1b[".to_vec();
         for i in 0..max_params + 5 { // Exceed max params
             seq_max_params.extend_from_slice(format!("{}", i+1).as_bytes());
             seq_max_params.push(b';');
         }
         seq_max_params.extend_from_slice(b"A"); // Final byte
         let commands_max_params = parse_bytes(&seq_max_params);
         assert_eq!(commands_max_params.len(), 1);
          // The parser should have processed up to MAX_CSI_PARAMS parameters.
          // CUU only uses the first parameter.
          if let AnsiCommand::Csi(CsiCommand::CursorUp(param)) = &commands_max_params[0] {
               assert_eq!(*param, 1, "First param should be 1, others truncated after max"); // Depends on parsing, but default 1 is likely
          } else {
               panic!("Expected CSI CUU command, got {:?}", commands_max_params[0]);
          }
     }


    #[test]
    fn test_csi_intermediates() {
        // Intermediate bytes are between parameters/private marker and the final byte.
        // They are typically ignored for dispatch but collected by the parser.
        let commands = parse_bytes(b"\x1b[?1!@A"); // Private marker ?, param 1, intermediates !, @, final A
        assert_eq!(commands.len(), 1);
         // Private modes are dispatched based on param(s) and final byte, intermediates ignored for dispatch
         assert_eq!(commands[0], AnsiCommand::Csi(CsiCommand::DecPrivateModeSet(DecPrivateMode::CursorKeysAppMode))); // ?1h -> DECCKM Set

         let commands_std_intermediate = parse_bytes(b"\x1b[1;2!?A"); // Param 1;2, intermediates !, ?, final A
         assert_eq!(commands_std_intermediate.len(), 1);
          // Standard CSI dispatch uses params and final byte, intermediates ignored
         assert_eq!(commands_std_intermediate[0], AnsiCommand::Csi(CsiCommand::CursorUp(1))); // CUU 1
    }


    #[test]
    fn test_csi_sgr_commands() {
        let commands = parse_bytes(b"\x1b[1;31;4;43m"); // Bold, Red FG, Underline, Yellow BG
        assert_eq!(commands.len(), 1);
        if let AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(params)) = &commands[0] {
            // Order of parameters in the Vec might not be guaranteed unless parser preserves it.
            // Let's check the set of parameters.
            assert_eq!(params.len(), 4);
            assert!(params.contains(&SgrParameter::Bold));
            assert!(params.contains(&SgrParameter::ForegroundBasic(1))); // 31 is Red
            assert!(params.contains(&SgrParameter::Underline));
            assert!(params.contains(&SgrParameter::BackgroundBasic(3))); // 43 is Yellow
        } else {
             panic!("Expected SGR command, got {:?}", commands[0]);
        }

        // Test 256-color and RGB
        let commands_extended = parse_bytes(b"\x1b[38;5;100m\x1b[48;2;10;20;30m");
        assert_eq!(commands_extended.len(), 2); // Each SGR sequence generates one command
        if let AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(params)) = &commands_extended[0] {
             assert_eq!(params.len(), 1);
             assert!(params.contains(&SgrParameter::Foreground256Color(100)));
        } else { panic!("Expected SGR command, got {:?}", commands_extended[0]); }
        if let AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(params)) = &commands_extended[1] {
             assert_eq!(params.len(), 1);
             assert!(params.contains(&SgrParameter::BackgroundRgb(10, 20, 30)));
        } else { panic!("Expected SGR command, got {:?}", commands_extended[1]); }
    }


    #[test]
    fn test_csi_dec_private_mode_commands() {
        // DECSET and DECRST dispatch multiple commands if multiple parameters are given.
        let commands = parse_bytes(b"\x1b[?1h\x1b[?6;25h\x1b[?1l\x1b[?1049l");
        assert_eq!(commands.len(), 5); // 1h (1 cmd), 6;25h (2 cmds), 1l (1 cmd), 1049l (1 cmd)
        assert_eq!(commands[0], AnsiCommand::Csi(CsiCommand::DecPrivateModeSet(DecPrivateMode::CursorKeysAppMode)));
        assert_eq!(commands[1], AnsiCommand::Csi(CsiCommand::DecPrivateModeSet(DecPrivateMode::OriginMode)));
        assert_eq!(commands[2], AnsiCommand::Csi(CsiCommand::DecPrivateModeSet(DecPrivateMode::CursorVisibility)));
        assert_eq!(commands[3], AnsiCommand::Csi(CsiCommand::DecPrivateModeReset(DecPrivateMode::CursorKeysAppMode)));
        assert_eq!(commands[4], AnsiCommand::Csi(CsiCommand::DecPrivateModeReset(DecPrivateMode::UseAltScreenAndSaveRestoreCursor)));
    }


    // --- OSC Sequence Tests ---

    #[test]
    fn test_osc_set_title_commands() {
        let commands_0 = parse_bytes(b"\x1b]0;My Terminal Title\x07");
        assert_eq!(commands_0.len(), 1);
        assert_eq!(commands_0[0], AnsiCommand::Osc(OscCommand::SetIconAndWindowTitle("0;My Terminal Title".to_string())));

        let commands_2 = parse_bytes(b"\x1b]2;Just the Title\x1b\\"); // Using ST terminator
        assert_eq!(commands_2.len(), 1);
        assert_eq!(commands_2[0], AnsiCommand::Osc(OscCommand::SetWindowTitle("2;Just the Title".to_string())));

         let commands_1 = parse_bytes(&[0x9d, b'1', b';', b'I', b'c', b'o', b'n', 0x9c]); // Using C1 introducer and terminator
         assert_eq!(commands_1.len(), 1);
         assert_eq!(commands_1[0], AnsiCommand::Osc(OscCommand::SetIconName("1;Icon".to_string())));
    }


    // --- Incremental Input / Buffering Tests ---

    #[test]
    fn test_incomplete_csi_sequence() {
        let commands1 = parse_bytes_incremental(b"\x1b[31"); // Incomplete SGR
        assert!(commands1.is_empty()); // No command yet

        let commands2 = parse_bytes_incremental(b";4m"); // Complete SGR
        assert_eq!(commands2.len(), 1);
        if let AnsiCommand::Csi(CsiCommand::SetGraphicsRendition(params)) = &commands2[0] {
            assert_eq!(params.len(), 2);
            assert!(params.contains(&SgrParameter::ForegroundBasic(1)));
            assert!(params.contains(&SgrParameter::Underline));
        } else { panic!("Expected SGR command, got {:?}", commands2[0]); }
    }

    #[test]
    fn test_incomplete_osc_sequence() {
        let commands1 = parse_bytes_incremental(b"\x1b]0;"); // Incomplete OSC
        assert!(commands1.is_empty()); // No command yet

        let commands2 = parse_bytes_incremental(b"short title"); // Add more string
        assert!(commands2.is_empty()); // Still incomplete

        let commands3 = parse_bytes_incremental(b"\x07"); // Complete OSC
        assert_eq!(commands3.len(), 1);
        assert_eq!(commands3[0], AnsiCommand::Osc(OscCommand::SetIconAndWindowTitle("0;short title".to_string())));
    }


    // --- Error Handling Tests ---

    #[test]
    fn test_csi_interrupted_by_esc() {
        let commands1 = parse_bytes_incremental(b"\x1b[1;"); // Start CSI
        assert!(commands1.is_empty());

        let commands2 = parse_bytes_incremental(b"\x1b"); // ESC interrupts
        assert_eq!(commands2.len(), 1);
        // Parser should emit an Error command for the aborted sequence
        if let AnsiCommand::Error(msg) = &commands2[0] {
            assert!(msg.contains("CSI sequence aborted by ESC"));
        } else {
             panic!("Expected Error command, got {:?}", commands2[0]);
        }
         // The ESC byte itself might be processed as the start of a new sequence,
         // but the Error command is the key output here for the aborted sequence.
    }

    #[test]
    fn test_osc_interrupted_by_esc() {
        let commands1 = parse_bytes_incremental(b"\x1b]0;tit"); // Start OSC
        assert!(commands1.is_empty());

        let commands2 = parse_bytes_incremental(b"\x1b"); // ESC interrupts
        // OSC interrupt by ESC does NOT emit an Error in the current parser,
        // it just aborts the string and transitions to Escape state.
        assert!(commands2.is_empty());
         // The ESC byte starts a new sequence, but no command related to the OSC is output.
    }

    #[test]
    fn test_unexpected_byte_in_csi_entry() {
        let commands = parse_bytes(b"\x1b[!"); // '!' is intermediate, not valid in CSIEntry after '['
        assert_eq!(commands.len(), 1);
        // Parser should emit an error and return to Ground.
         if let AnsiCommand::Error(msg) = &commands[0] {
             assert!(msg.contains("Unexpected byte 0x21 in CSIEntry state")); // 0x21 is '!'
         } else {
              panic!("Expected Error command, got {:?}", commands[0]);
         }
    }

    #[test]
    fn test_unexpected_byte_in_csi_param() {
        let commands = parse_bytes(b"\x1b[1;A"); // 'A' is a final byte, not valid as parameter byte after ';'
        assert_eq!(commands.len(), 1);
         // Parser should emit an error and return to Ground.
         if let AnsiCommand::Error(msg) = &commands[0] {
             assert!(msg.contains("Unexpected byte 0x41 in CSIParam state")); // 0x41 is 'A'
         } else {
              panic!("Expected Error command, got {:?}", commands[0]);
         }
    }


    // --- UTF-8 Handling Tests ---

    #[test]
    fn test_utf8_multi_byte() {
        let commands = parse_bytes("你好".as_bytes()); // "Nǐ hǎo"
        assert_eq!(commands.len(), 2);
        assert_eq!(commands, vec![
            AnsiCommand::Print('你'),
            AnsiCommand::Print('好'),
        ]);
    }

    #[test]
    fn test_utf8_split_across_feeds() {
        let mut parser = AnsiParser::new();
        let bytes = "你好".as_bytes();

        let commands1 = parser.feed_byte(bytes[0]); // Part of '你'
        assert!(commands1.is_empty());

        let commands2 = parser.feed_byte(bytes[1]); // Part of '你'
        assert!(commands2.is_empty());

        let commands3 = parser.feed_byte(bytes[2]); // Completes '你'
        assert_eq!(commands3.len(), 1);
        assert_eq!(commands3[0], AnsiCommand::Print('你'));

        let commands4 = parser.feed_slice(&bytes[3..]); // Completes '好'
        assert_eq!(commands4.len(), 1);
        assert_eq!(commands4[0], AnsiCommand::Print('好'));
    }

     #[test]
     fn test_utf8_split_interrupted_by_esc() {
         let mut parser = AnsiParser::new();
         let bytes = "你好".as_bytes();

         let commands1 = parser.feed_byte(bytes[0]); // Part of '你'
         assert!(commands1.is_empty());

         let commands2 = parser.feed_slice(b"\x1b["); // ESC [ interrupts UTF-8
         // When UTF-8 decoding is interrupted, the parser should emit a REPLACEMENT_CHARACTER
         assert_eq!(commands2.len(), 1);
         assert_eq!(commands2[0], AnsiCommand::Print(std::char::REPLACEMENT_CHARACTER));
         // The ESC [ also starts a new sequence, handled internally by the parser state

         let commands3 = parser.feed_slice(b"2J"); // Complete the CSI sequence
         assert_eq!(commands3.len(), 1);
         assert_eq!(commands3[0], AnsiCommand::Csi(CsiCommand::EraseDisplay(EraseMode::All)));
     }


     #[test]
     fn test_invalid_utf8_sequence() {
         let commands = parse_bytes(b"A\x80B"); // Lone continuation byte
         assert_eq!(commands.len(), 3);
         assert_eq!(commands, vec![
             AnsiCommand::Print('A'),
             AnsiCommand::Print(std::char::REPLACEMENT_CHARACTER),
             AnsiCommand::Print('B'),
         ]);
     }

     #[test]
     fn test_invalid_utf8_sequence_mid() {
         let commands = parse_bytes(b"A\xc3\x41B"); // Start of 2-byte (\xc3) followed by invalid continuation (\x41)
         assert_eq!(commands.len(), 3);
         assert_eq!(commands, vec![
              AnsiCommand::Print('A'),
              AnsiCommand::Print(std::char::REPLACEMENT_CHARACTER), // \xc3\x41 is invalid UTF-8
              AnsiCommand::Print('B'),
         ]);
     }

      #[test]
     fn test_invalid_utf8_within_osc_string_lossy() {
          // UTF-8 parse error within OSC should result in replacement char in the string
          // when converted to String using from_utf8_lossy for the OSC command.
          let commands = parse_bytes(b"\x1b]0;Ti\xc3tle\x07"); // Incomplete UTF-8 \xc3 in string
          assert_eq!(commands.len(), 1);
          assert_eq!(commands[0], AnsiCommand::Osc(OscCommand::SetIconAndWindowTitle("0;Ti\u{fffd}tle".to_string())));

          let commands_c1_in_osc = parse_bytes(b"\x1b]0;Ti\x80tle\x07"); // C1 control in string
          assert_eq!(commands_c1_in_osc.len(), 1);
           // C1 controls (0x80-0x9F) should be ignored within OSC strings.
          assert_eq!(commands_c1_in_osc[0], AnsiCommand::Osc(OscCommand::SetIconAndWindowTitle("0;Title".to_string())));
      }
}
