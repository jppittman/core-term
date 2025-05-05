// src/ansi/lexer.rs

use log::{debug, trace, warn}; // Import logging macros

// --- Constants for Control Codes ---
// C0 Controls (0x00 - 0x1F, 0x7F)
const BEL: u8 = 0x07; // Bell
pub(super) const ESC: u8 = 0x1B; // Escape

// C1 Control Codes (0x80 - 0x9F)
// These can be represented as ESC + byte or single byte in 8-bit environments.
// Making these public within the crate::ansi module as they are used by the parser.
pub const IND: u8 = 0x84; // ESC D - Index
pub const NEL: u8 = 0x85; // ESC E - Next Line
pub const HTS: u8 = 0x88; // ESC H - Horizontal Tabulation Set
pub const RI: u8 = 0x8D;  // ESC M - Reverse Index
pub const SS2: u8 = 0x8E; // ESC N - Single Shift Select of G2 character set
pub const SS3: u8 = 0x8F; // ESC O - Single Shift Select of G3 character set
pub(super) const DCS: u8 = 0x90; // ESC P - Device Control String Introducer
pub(super) const CSI: u8 = 0x9B; // ESC [ - Control Sequence Introducer
pub const ST: u8 = 0x9C;  // ESC \ - String Terminator
pub(super) const OSC: u8 = 0x9D; // ESC ] - Operating System Command Introducer
pub(super) const PM: u8 = 0x9E;  // ESC ^ - Privacy Message Introducer
pub(super) const APC: u8 = 0x9F; // ESC _ - Application Program Command Introducer

// Placeholder C1 codes for specific single-character ESC sequences handled by the parser.
// Using currently unused C0 codes as placeholders avoids conflict with actual C1 codes.
// WHY: This mapping allows the lexer to emit a single C1Control token for these
// common sequences, simplifying the parser's Escape state handling.
/// Placeholder for ESC 7 (DECSC - Save Cursor). Uses ETB (0x17).
pub(super) const ESC_7: u8 = 0x17;
/// Placeholder for ESC 8 (DECRC - Restore Cursor). Uses CAN (0x18).
pub(super) const ESC_8: u8 = 0x18;
/// Placeholder for ESC c (RIS - Reset to Initial State). Uses EM (0x19).
pub(super) const ESC_C: u8 = 0x19;


/// Represents the different states of the ANSI lexer state machine.
///
/// The lexer transitions between these states based on the input byte stream
/// to correctly identify and tokenize ANSI escape sequences and control codes.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum LexerState { // Made pub for potential use by parser/tests
    /// Initial state, processing normal characters and C0 controls. Expects printable chars, C0, ESC, or C1.
    Ground,
    /// Received an Escape character (0x1B). Expects sequence introducer ([, P, ], _, ^, etc.) or other ESC sequence bytes.
    Escape,
    /// Received an Escape followed by a byte that introduces a 2-byte sequence (Fe bytes 0x20-0x2F). Expects final byte (0x30-0x7E).
    TwoByteEscape,
    /// Received CSI (ESC [ or 0x9B). Expects parameters (0x30-0x3F), intermediates (0x20-0x2F), or final byte (0x40-0x7E).
    CsiEntry,
    /// Inside a Control Sequence (CSI), processing parameters (0x30-0x3F). Expects more params, intermediates, or final byte.
    CsiParam,
    /// Inside a Control Sequence (CSI), processing intermediate bytes (0x20-0x2F). Expects more intermediates or final byte.
    CsiIntermediate,
    /// Received DCS (ESC P or 0x90). Expects parameters, intermediates, or data string start.
    DcsEntry,
    /// Inside a Device Control String (DCS), processing parameters (0x30-0x3F).
    DcsParam,
    /// Inside a Device Control String (DCS), processing intermediate bytes (0x20-0x2F).
    DcsIntermediate,
    /// Inside a Device Control String (DCS), processing the data string until ST (ESC \ or 0x9C).
    DcsPassThrough,
    /// Received OSC (ESC ] or 0x9D). Processing the string until ST or BEL (0x07).
    OscString,
    /// Received PM (ESC ^ or 0x9E). Processing the string until ST.
    PmString,
    /// Received APC (ESC _ or 0x9F). Processing the string until ST.
    ApcString,
}

/// Represents a token emitted by the ANSI lexer.
///
/// These tokens abstract the raw byte stream into meaningful units for the parser.
#[derive(Debug, PartialEq, Clone)]
pub enum AnsiToken {
    /// A single printable character.
    Print(char),
    /// A C0 control character byte (0x00-0x1F, 0x7F). The parser interprets these.
    C0Control(u8),
    /// A C1 control character byte (0x80-0x9F) OR a placeholder for specific ESC sequences (like ESC_7).
    C1Control(u8),
    /// A byte following an Escape that introduces a 2-byte sequence (e.g., for SCS). The byte itself is the intermediate.
    TwoByteEscape(u8),
    /// Signals entry into a CSI sequence (parameters/intermediates/final byte follow).
    CsiEntry,
    /// A parameter byte within a CSI or DCS sequence (0x30-0x3F). Includes digits, ';', and private markers '?', '<', '=', '>'.
    CsiParam(u8),
    /// An intermediate byte within a CSI or DCS sequence (0x20-0x2F).
    CsiIntermediate(u8),
    /// The final byte of a CSI sequence (0x40-0x7E), determining the command.
    CsiFinal(u8),
    /// Signals entry into a DCS sequence (parameters/intermediates/data string follow).
    DcsEntry,
    /// A parameter byte within a DCS sequence (0x30-0x3F).
    DcsParam(u8),
    /// An intermediate byte within a DCS sequence (0x20-0x2F).
    DcsIntermediate(u8),
    /// A data byte within the DCS string.
    DcsStringByte(u8),
    /// Signals entry into an OSC sequence (string bytes follow).
    OscString,
    /// A data byte within the OSC string.
    OscStringByte(u8),
    /// Signals entry into a PM sequence (string bytes follow).
    PmString,
    /// A data byte within the PM string.
    PmStringByte(u8),
    /// Signals entry into an APC sequence (string bytes follow).
    ApcString,
    /// A data byte within the APC string.
    ApcStringByte(u8),
    /// The String Terminator byte sequence (0x9C or ESC \). Signals the end of DCS, OSC, PM, APC.
    StringTerminator,
    /// A byte that was ignored according to ANSI rules (e.g., C0 within certain strings).
    Ignored(u8),
    /// A byte that is not recognized in the current state, forms an invalid sequence, or indicates a lexer error.
    Error(u8),
}

/// A lexer that tokenizes ANSI escape sequences and control characters from a byte stream.
///
/// It maintains state between calls to `process_byte`, allowing for fragmented sequences
/// to be processed correctly across multiple reads. Use `take_tokens` to retrieve the
/// generated tokens for parsing.
pub struct AnsiLexer {
    /// The current state of the lexer's state machine.
    state: LexerState,
    /// A buffer holding tokens generated since the last `take_tokens` call.
    tokens: Vec<AnsiToken>,
}

impl AnsiLexer {
    /// Creates a new `AnsiLexer` in the `Ground` state.
    pub fn new() -> Self {
        AnsiLexer {
            state: LexerState::Ground,
            tokens: Vec::new(),
        }
    }

    /// Processes a single byte, updating the lexer's state and potentially emitting tokens.
    ///
    /// # Arguments
    /// * `byte`: The input byte to process.
    pub fn process_byte(&mut self, byte: u8) {
        trace!("Lexer Processing byte: {:02X} ('{}') in state: {:?}",
               byte, if byte.is_ascii_graphic() || byte == b' ' { byte as char } else { '.' }, self.state);
        let initial_state = self.state;

        // Delegate handling to the method corresponding to the current state.
        match initial_state {
            LexerState::Ground => self.handle_ground(byte),
            LexerState::Escape => self.handle_escape(byte),
            LexerState::TwoByteEscape => self.handle_two_byte_escape(byte),
            LexerState::CsiEntry => self.handle_csi_entry(byte),
            LexerState::CsiParam => self.handle_csi_param(byte),
            LexerState::CsiIntermediate => self.handle_csi_intermediate(byte),
            LexerState::DcsEntry => self.handle_dcs_entry(byte),
            LexerState::DcsParam => self.handle_dcs_param(byte),
            LexerState::DcsIntermediate => self.handle_dcs_intermediate(byte),
            LexerState::DcsPassThrough => self.handle_dcs_passthrough(byte),
            LexerState::OscString => self.handle_osc_string(byte),
            LexerState::PmString => self.handle_pm_string(byte),
            LexerState::ApcString => self.handle_apc_string(byte),
        }

        // Log state transitions for debugging purposes.
        if self.state != initial_state {
             trace!("Lexer State transition: {:?} -> {:?}", initial_state, self.state);
        }
    }

    /// Returns the list of tokens generated since the last call and clears the internal buffer.
    pub fn take_tokens(&mut self) -> Vec<AnsiToken> {
        std::mem::take(&mut self.tokens)
    }

    /// Pushes a token to the internal token list.
    fn push_token(&mut self, token: AnsiToken) {
        debug!("Lexer Pushed token: {:?}", token);
        self.tokens.push(token);
    }

    /// Resets the lexer state to `Ground`. Called when a sequence completes or aborts.
    fn reset_state(&mut self) {
        self.state = LexerState::Ground;
    }

    /// Handles bytes received while the lexer is in the `Ground` state.
    fn handle_ground(&mut self, byte: u8) {
        match byte {
            // C0 Control Codes (excluding ESC) and DEL (0x7F).
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                self.push_token(AnsiToken::C0Control(byte));
            }
            ESC => { // Escape character starts an escape sequence.
                self.state = LexerState::Escape;
            }
            // C1 Control Codes (0x80-0x9F).
            0x80..=0x9F => {
                // Handle C1 introducers directly by changing state and emitting entry tokens.
                // Other C1 codes are emitted as C1Control tokens.
                match byte {
                    CSI => {
                        self.push_token(AnsiToken::CsiEntry);
                        self.state = LexerState::CsiEntry;
                    }
                    DCS => {
                        self.push_token(AnsiToken::DcsEntry);
                        self.state = LexerState::DcsEntry;
                    }
                    OSC => {
                        self.push_token(AnsiToken::OscString);
                        self.state = LexerState::OscString;
                    }
                    PM => {
                        self.push_token(AnsiToken::PmString);
                        self.state = LexerState::PmString;
                    }
                    APC => {
                        self.push_token(AnsiToken::ApcString);
                        self.state = LexerState::ApcString;
                    }
                    ST => {
                        // ST is only expected after DCS/OSC/PM/APC. In Ground, it's unexpected.
                        warn!("Unexpected ST (0x9C) received in Ground state.");
                        self.push_token(AnsiToken::StringTerminator); // Still tokenize it
                    }
                    _ => self.push_token(AnsiToken::C1Control(byte)), // Emit other C1 controls directly.
                }
            }
            // Printable characters (including 8-bit range if not UTF-8 mode, though UTF-8 handling is preferred).
            _ => {
                // Attempt to convert the byte to a char. This handles ASCII directly.
                // For multi-byte UTF-8, the terminal/parser layer should handle decoding.
                // The lexer's job here is primarily to distinguish controls/escapes from printables.
                if let Some(c) = std::char::from_u32(byte as u32) {
                    self.push_token(AnsiToken::Print(c));
                } else {
                    // This case might occur for invalid bytes in certain encodings.
                    warn!("Received byte {:02X} that couldn't be converted to char in Ground state", byte);
                    self.push_token(AnsiToken::Error(byte));
                }
            }
        }
    }


    /// Handles bytes received while the lexer is in the `Escape` state (after ESC).
    fn handle_escape(&mut self, byte: u8) {
        match byte {
            // C0 Controls (excluding ESC) and DEL - Execute immediately, abort ESC sequence.
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                warn!("ESC sequence aborted by C0 control: 0x{:02X}", byte);
                self.push_token(AnsiToken::C0Control(byte)); // Process the C0 code.
                self.reset_state();
            }
            ESC => { // ESC ESC - Treat the second ESC as starting a new sequence. Remain in Escape state.
                warn!("Received ESC ESC sequence.");
                // No token emitted here, just stay in Escape state.
            }
            // Sequence Introducers
            b'[' => { // CSI
                self.push_token(AnsiToken::CsiEntry);
                self.state = LexerState::CsiEntry;
            }
            b']' => { // OSC
                self.push_token(AnsiToken::OscString);
                self.state = LexerState::OscString;
            }
            b'P' => { // DCS
                self.push_token(AnsiToken::DcsEntry);
                self.state = LexerState::DcsEntry;
            }
            b'^' => { // PM
                self.push_token(AnsiToken::PmString);
                self.state = LexerState::PmString;
            }
            b'_' => { // APC
                self.push_token(AnsiToken::ApcString);
                self.state = LexerState::ApcString;
            }
            b'\\' => { // ST (String Terminator)
                self.push_token(AnsiToken::StringTerminator);
                self.reset_state();
            }
            // Intermediate bytes for 2-byte sequences (Fe: 0x20-0x2F), e.g., SCS.
            0x20..=0x2F => {
                self.push_token(AnsiToken::TwoByteEscape(byte));
                self.state = LexerState::TwoByteEscape; // Expect one more byte (finalizer).
            }
            // Final bytes for 2-byte sequences (Fs: 0x30-0x7E) or single-char sequences.
            0x30..=0x7E => {
                // These complete a single-character escape sequence (like ESC D, ESC c)
                // or are invalid after ESC. Map known ones to C1 placeholders.
                let c1_equiv = match byte {
                    b'D' => Some(IND), b'E' => Some(NEL), b'H' => Some(HTS),
                    b'M' => Some(RI), b'N' => Some(SS2), b'O' => Some(SS3),
                    // Map special ESC sequences to placeholder C1 codes for parser convenience.
                    b'7' => Some(ESC_7), // DECSC
                    b'8' => Some(ESC_8), // DECRC
                    b'c' => Some(ESC_C), // RIS
                    _ => None,
                };
                if let Some(c1) = c1_equiv {
                    self.push_token(AnsiToken::C1Control(c1));
                } else {
                    // If not a known single-char sequence (like ESC ( B), the intermediate
                    // byte 0x28 would have transitioned to TwoByteEscape state first).
                    // An unknown ESC + Fs is typically ignored or treated as an error.
                    warn!("Unsupported escape sequence: ESC {:02X} ('{}')", byte, if byte.is_ascii() { byte as char } else { '.' });
                    self.push_token(AnsiToken::Error(byte));
                }
                self.reset_state(); // Sequence finished or errored.
            }
            // C1 Controls (0x80-0x9F) - Execute immediately, abort ESC sequence.
            0x80..=0x9F => {
                 warn!("ESC sequence aborted by C1 control: 0x{:02X}", byte);
                 self.push_token(AnsiToken::C1Control(byte)); // Process the C1 code.
                 self.reset_state();
            }
            // Any other byte is unexpected after ESC.
             _ => {
                 warn!("Unexpected byte 0x{:02X} following ESC", byte);
                 self.push_token(AnsiToken::Error(byte));
                 self.reset_state();
            }
        }
    }

    /// Handles bytes in the `TwoByteEscape` state (after ESC and an intermediate byte 0x20-0x2F).
    /// Expects a final byte (0x30-0x7E) to complete sequences like SCS.
    fn handle_two_byte_escape(&mut self, byte: u8) {
        match byte {
            // C0 Controls (excluding ESC) and DEL - Execute, abort sequence.
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                warn!("Two-byte ESC sequence aborted by C0 control: 0x{:02X}", byte);
                self.push_token(AnsiToken::C0Control(byte));
                self.reset_state();
            }
            ESC => { // Abort on ESC.
                warn!("Two-byte ESC sequence aborted by second ESC");
                self.push_token(AnsiCommand::Error(ESC)); // Report the aborting ESC as error.
                self.reset_state();
            }
            // Final bytes for 2-byte sequences (0x30-0x7E). e.g., 'B' in ESC ( B
            0x30..=0x7E => {
                // The parser needs both the intermediate (from the TwoByteEscape token)
                // and this final byte to interpret the command (e.g., SCS).
                // We emit a C1Control token containing the *final* byte. The parser
                // must check if the *previous* token was TwoByteEscape to handle it correctly.
                // This approach keeps the lexer simpler but puts more logic in the parser.
                self.push_token(AnsiToken::C1Control(byte)); // Parser needs context from previous token.
                self.reset_state();
            }
            // C1 Controls - Execute, abort sequence.
            0x80..=0x9F => {
                 warn!("Two-byte ESC sequence aborted by C1 control: 0x{:02X}", byte);
                 self.push_token(AnsiToken::C1Control(byte));
                 self.reset_state();
            }
            // Intermediate bytes (0x20-0x2F) - Invalid here, sequence expects a final byte.
            0x20..=0x2F => {
                warn!("Unexpected intermediate byte {:02X} in TwoByteEscape state", byte);
                self.push_token(AnsiToken::Error(byte));
                self.reset_state();
            }
            // Any other byte is unexpected.
             _ => {
                 warn!("Unexpected byte 0x{:02X} in TwoByteEscape state", byte);
                 self.push_token(AnsiToken::Error(byte));
                 self.reset_state();
            }
        }
    }


    /// Handles bytes in the `CsiEntry` state (after ESC [ or C1 CSI).
    /// Expects parameters, intermediates, or a final byte.
    fn handle_csi_entry(&mut self, byte: u8) {
        match byte {
            // C0 Controls (excluding ESC) and DEL - Execute, abort CSI.
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                warn!("CSI sequence aborted by C0 control: 0x{:02X}", byte);
                self.push_token(AnsiToken::C0Control(byte));
                self.reset_state();
            }
            ESC => { // Abort on ESC.
                warn!("CSI sequence aborted by ESC");
                self.push_token(AnsiCommand::Error(ESC)); // Report the aborting ESC as error.
                self.reset_state();
            }
            // Parameter bytes (0x30-0x3F). Includes digits, ';', and private markers '?', '<', '=', '>'.
            0x30..=0x3F => {
                self.push_token(AnsiToken::CsiParam(byte));
                self.state = LexerState::CsiParam; // Transition to parameter state.
            }
            // Intermediate bytes (0x20-0x2F).
            0x20..=0x2F => {
                self.push_token(AnsiToken::CsiIntermediate(byte));
                self.state = LexerState::CsiIntermediate; // Transition to intermediate state.
            }
            // Final bytes (0x40-0x7E). Completes the CSI sequence.
            0x40..=0x7E => {
                self.push_token(AnsiToken::CsiFinal(byte));
                self.reset_state(); // Sequence finished.
            }
            // C1 Controls - Execute, abort CSI.
            0x80..=0x9F => {
                 warn!("CSI sequence aborted by C1 control: 0x{:02X}", byte);
                 self.push_token(AnsiToken::C1Control(byte));
                 self.reset_state();
            }
            // Any other byte is unexpected.
             _ => {
                 warn!("Unexpected byte 0x{:02X} in CsiEntry state", byte);
                 self.push_token(AnsiCommand::Error(byte));
                 self.reset_state();
            }
        }
    }

    /// Handles bytes in the `CsiParam` state.
    /// Expects more parameters, intermediates, or a final byte.
    fn handle_csi_param(&mut self, byte: u8) {
        match byte {
            // C0 Controls (excluding ESC) and DEL - Execute, abort CSI.
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                warn!("CSI sequence aborted by C0 control: 0x{:02X}", byte);
                self.push_token(AnsiToken::C0Control(byte));
                self.reset_state();
            }
            ESC => { // Abort on ESC.
                warn!("CSI sequence aborted by ESC");
                self.push_token(AnsiCommand::Error(ESC));
                self.reset_state();
            }
            // Parameter bytes (0x30-0x3F).
            0x30..=0x3F => {
                self.push_token(AnsiToken::CsiParam(byte));
                // Stay in CsiParam state.
            }
            // Intermediate bytes (0x20-0x2F).
            0x20..=0x2F => {
                self.push_token(AnsiToken::CsiIntermediate(byte));
                self.state = LexerState::CsiIntermediate; // Transition to intermediate state.
            }
            // Final bytes (0x40-0x7E). Completes the CSI sequence.
            0x40..=0x7E => {
                self.push_token(AnsiToken::CsiFinal(byte));
                self.reset_state(); // Sequence finished.
            }
            // C1 Controls - Execute, abort CSI.
            0x80..=0x9F => {
                 warn!("CSI sequence aborted by C1 control: 0x{:02X}", byte);
                 self.push_token(AnsiToken::C1Control(byte));
                 self.reset_state();
            }
            // Any other byte is unexpected.
            _ => {
                 warn!("Unexpected byte 0x{:02X} in CsiParam state", byte);
                 self.push_token(AnsiCommand::Error(byte));
                 self.reset_state();
            }
        }
    }

    /// Handles bytes in the `CsiIntermediate` state.
    /// Expects more intermediates or a final byte.
    fn handle_csi_intermediate(&mut self, byte: u8) {
         match byte {
            // C0 Controls (excluding ESC) and DEL - Execute, abort CSI.
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                warn!("CSI sequence aborted by C0 control: 0x{:02X}", byte);
                self.push_token(AnsiToken::C0Control(byte));
                self.reset_state();
            }
            ESC => { // Abort on ESC.
                warn!("CSI sequence aborted by ESC");
                self.push_token(AnsiCommand::Error(ESC));
                self.reset_state();
            }
            // Intermediate bytes (0x20-0x2F).
            0x20..=0x2F => {
                self.push_token(AnsiToken::CsiIntermediate(byte));
                // Stay in CsiIntermediate state.
            }
            // Final bytes (0x40-0x7E). Completes the CSI sequence.
            0x40..=0x7E => {
                self.push_token(AnsiToken::CsiFinal(byte));
                self.reset_state(); // Sequence finished.
            }
            // C1 Controls - Execute, abort CSI.
            0x80..=0x9F => {
                 warn!("CSI sequence aborted by C1 control: 0x{:02X}", byte);
                 self.push_token(AnsiToken::C1Control(byte));
                 self.reset_state();
            }
            // Parameter bytes (0x30-0x3F) - Invalid here, abort.
            0x30..=0x3F => {
                warn!("Unexpected parameter byte {:02X} in CsiIntermediate state", byte);
                self.push_token(AnsiCommand::Error(byte));
                self.reset_state();
            }
            // Any other byte is unexpected.
             _ => {
                 warn!("Unexpected byte 0x{:02X} in CsiIntermediate state", byte);
                 self.push_token(AnsiCommand::Error(byte));
                 self.reset_state();
            }
        }
    }

    /// Handles bytes in the `DcsEntry` state (after ESC P or C1 DCS).
    /// Expects parameters, intermediates, or the start of the data string.
    fn handle_dcs_entry(&mut self, byte: u8) {
        match byte {
            // C0 Controls (excluding ESC) and DEL - Ignore according to spec.
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                trace!("Ignoring C0 control 0x{:02X} in DcsEntry state", byte);
                self.push_token(AnsiToken::Ignored(byte));
            }
            ESC => { // Abort on ESC.
                warn!("DCS sequence aborted by ESC");
                // Don't push Error(ESC) immediately, transition to Escape state
                // to correctly handle potential ST (ESC \).
                self.state = LexerState::Escape;
            }
            // Parameter bytes (0x30-0x3F).
            0x30..=0x3F => {
                self.push_token(AnsiToken::DcsParam(byte));
                self.state = LexerState::DcsParam;
            }
            // Intermediate bytes (0x20-0x2F).
            0x20..=0x2F => {
                self.push_token(AnsiToken::DcsIntermediate(byte));
                self.state = LexerState::DcsIntermediate;
            }
            // Data string starts (Final bytes in CSI range 0x40-0x7E are data here).
            0x40..=0x7E => {
                self.push_token(AnsiToken::DcsStringByte(byte));
                self.state = LexerState::DcsPassThrough; // Enter data collection state.
            }
            // C1 Controls (excluding ST) - Ignore? Or abort? Let's ignore.
            0x80..=0x9B | 0x9D..=0x9F => {
                 trace!("Ignoring C1 control 0x{:02X} in DcsEntry state", byte);
                 self.push_token(AnsiToken::Ignored(byte));
            }
            ST => { // ST (String Terminator) terminates immediately, even if empty.
                 self.push_token(AnsiToken::StringTerminator);
                 self.reset_state();
            }
            // Any other byte is unexpected.
             _ => {
                 warn!("Unexpected byte 0x{:02X} in DcsEntry state", byte);
                 self.push_token(AnsiCommand::Error(byte));
                 self.reset_state();
            }
        }
    }

    /// Handles bytes in the `DcsParam` state.
    /// Expects more parameters, intermediates, or the start of the data string.
    fn handle_dcs_param(&mut self, byte: u8) {
        match byte {
            // C0 Controls (excluding ESC) and DEL - Ignore.
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                trace!("Ignoring C0 control 0x{:02X} in DcsParam state", byte);
                self.push_token(AnsiToken::Ignored(byte));
            }
            ESC => { // Abort on ESC.
                warn!("DCS sequence aborted by ESC");
                self.state = LexerState::Escape;
            }
            // Parameter bytes (0x30-0x3F).
            0x30..=0x3F => {
                self.push_token(AnsiToken::DcsParam(byte));
                // Stay in DcsParam state.
            }
            // Intermediate bytes (0x20-0x2F).
            0x20..=0x2F => {
                self.push_token(AnsiToken::DcsIntermediate(byte));
                self.state = LexerState::DcsIntermediate;
            }
            // Data string starts (0x40-0x7E).
            0x40..=0x7E => {
                 self.push_token(AnsiToken::DcsStringByte(byte));
                 self.state = LexerState::DcsPassThrough;
            }
            // C1 Controls (excluding ST) - Ignore.
             0x80..=0x9B | 0x9D..=0x9F => {
                  trace!("Ignoring C1 control 0x{:02X} in DcsParam state", byte);
                  self.push_token(AnsiToken::Ignored(byte));
             }
             ST => { // ST terminates immediately.
                  self.push_token(AnsiToken::StringTerminator);
                  self.reset_state();
             }
            // Any other byte is unexpected.
             _ => {
                 warn!("Unexpected byte 0x{:02X} in DcsParam state", byte);
                 self.push_token(AnsiCommand::Error(byte));
                 self.reset_state();
            }
        }
    }

    /// Handles bytes in the `DcsIntermediate` state.
    /// Expects more intermediates or the start of the data string.
    fn handle_dcs_intermediate(&mut self, byte: u8) {
         match byte {
            // C0 Controls (excluding ESC) and DEL - Ignore.
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                trace!("Ignoring C0 control 0x{:02X} in DcsIntermediate state", byte);
                self.push_token(AnsiToken::Ignored(byte));
            }
            ESC => { // Abort on ESC.
                warn!("DCS sequence aborted by ESC");
                self.state = LexerState::Escape;
            }
            // Intermediate bytes (0x20-0x2F).
            0x20..=0x2F => {
                self.push_token(AnsiToken::DcsIntermediate(byte));
                // Stay in DcsIntermediate state.
            }
            // Data string starts (0x40-0x7E).
            0x40..=0x7E => {
                 self.push_token(AnsiToken::DcsStringByte(byte));
                 self.state = LexerState::DcsPassThrough;
            }
            // C1 Controls (excluding ST) - Ignore.
             0x80..=0x9B | 0x9D..=0x9F => {
                  trace!("Ignoring C1 control 0x{:02X} in DcsIntermediate state", byte);
                  self.push_token(AnsiToken::Ignored(byte));
             }
             ST => { // ST terminates immediately.
                  self.push_token(AnsiToken::StringTerminator);
                  self.reset_state();
             }
            // Parameter bytes (0x30-0x3F) - Invalid here, abort.
            0x30..=0x3F => {
                warn!("Unexpected parameter byte {:02X} in DcsIntermediate state", byte);
                self.push_token(AnsiCommand::Error(byte));
                self.reset_state();
            }
            // Any other byte is unexpected.
             _ => {
                 warn!("Unexpected byte 0x{:02X} in DcsIntermediate state", byte);
                 self.push_token(AnsiCommand::Error(byte));
                 self.reset_state();
            }
        }
    }

    /// Handles bytes in the `DcsPassThrough` state (collecting the data string).
    /// Expects data bytes or a String Terminator (ST or ESC \).
    fn handle_dcs_passthrough(&mut self, byte: u8) {
        match byte {
            // C0 Controls (excluding ESC) and DEL - Ignore.
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                trace!("Ignoring C0 control 0x{:02X} in DcsPassThrough state", byte);
                self.push_token(AnsiToken::Ignored(byte));
            }
            ESC => { // Could be start of ESC \ (ST) or an aborting ESC.
                self.state = LexerState::Escape; // Tentatively enter Escape state to check next byte.
            }
            ST => { // ST (String Terminator) C1 code.
                self.push_token(AnsiToken::StringTerminator);
                self.reset_state();
            }
            // C1 Controls (excluding ST) - Ignore.
            0x80..=0x9B | 0x9D..=0x9F => {
                 trace!("Ignoring C1 control 0x{:02X} in DcsPassThrough state", byte);
                 self.push_token(AnsiToken::Ignored(byte));
            }
            // Any other byte is part of the data string.
            _ => {
                self.push_token(AnsiToken::DcsStringByte(byte));
                // Stay in DcsPassThrough state.
            }
        }
    }

    /// Handles bytes in the `OscString` state.
    /// Expects string bytes or a terminator (BEL or ST).
    fn handle_osc_string(&mut self, byte: u8) {
        match byte {
            BEL => { // BEL (0x07) terminates OSC.
                self.push_token(AnsiToken::C0Control(BEL)); // Tokenize BEL itself for the parser.
                self.reset_state();
            }
            ST => { // ST (0x9C) also terminates OSC.
                self.push_token(AnsiToken::StringTerminator);
                self.reset_state();
            }
            // C0 Controls (excluding BEL, ESC) and DEL - Ignore within OSC string.
            0x00..=0x06 | 0x08..=0x1A | 0x1C..=0x1F | 0x7F => {
                trace!("Ignoring C0 control 0x{:02X} in OscString state", byte);
                self.push_token(AnsiToken::Ignored(byte));
            }
            ESC => { // Could be start of ESC \ (ST) or an aborting ESC.
                self.state = LexerState::Escape; // Tentatively enter Escape state.
            }
            // C1 Controls (excluding ST) - Ignore within OSC string.
            0x80..=0x9B | 0x9D..=0x9F => {
                 trace!("Ignoring C1 control 0x{:02X} in OscString state", byte);
                 self.push_token(AnsiToken::Ignored(byte));
            }
            // Any other byte is part of the OSC string.
            _ => {
                self.push_token(AnsiToken::OscStringByte(byte));
                // Stay in OscString state.
            }
        }
    }

    /// Handles bytes in the `PmString` state.
    /// Expects string bytes or a String Terminator (ST).
    fn handle_pm_string(&mut self, byte: u8) {
         match byte {
            ST => { // ST (String Terminator) C1 code.
                self.push_token(AnsiToken::StringTerminator);
                self.reset_state();
            }
            // C0 Controls (excluding ESC) and DEL - Ignore within PM string.
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                trace!("Ignoring C0 control 0x{:02X} in PmString state", byte);
                self.push_token(AnsiToken::Ignored(byte));
            }
            ESC => { // Could be start of ESC \ (ST) or an aborting ESC.
                self.state = LexerState::Escape; // Tentatively enter Escape state.
            }
            // C1 Controls (excluding ST) - Ignore within PM string.
            0x80..=0x9B | 0x9D..=0x9F => {
                 trace!("Ignoring C1 control 0x{:02X} in PmString state", byte);
                 self.push_token(AnsiToken::Ignored(byte));
            }
            // Any other byte is part of the PM string.
            _ => {
                self.push_token(AnsiToken::PmStringByte(byte));
                // Stay in PmString state.
            }
        }
    }

    /// Handles bytes in the `ApcString` state.
    /// Expects string bytes or a String Terminator (ST).
    fn handle_apc_string(&mut self, byte: u8) {
         match byte {
            ST => { // ST (String Terminator) C1 code.
                self.push_token(AnsiToken::StringTerminator);
                self.reset_state();
            }
            // C0 Controls (excluding ESC) and DEL - Ignore within APC string.
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                trace!("Ignoring C0 control 0x{:02X} in ApcString state", byte);
                self.push_token(AnsiToken::Ignored(byte));
            }
            ESC => { // Could be start of ESC \ (ST) or an aborting ESC.
                self.state = LexerState::Escape; // Tentatively enter Escape state.
            }
            // C1 Controls (excluding ST) - Ignore within APC string.
            0x80..=0x9B | 0x9D..=0x9F => {
                 trace!("Ignoring C1 control 0x{:02X} in ApcString state", byte);
                 self.push_token(AnsiToken::Ignored(byte));
            }
            // Any other byte is part of the APC string.
            _ => {
                self.push_token(AnsiToken::ApcStringByte(byte));
                // Stay in ApcString state.
            }
        }
    }
}

