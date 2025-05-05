// src/ansi/parser.rs

// Import logging macros
use log::{debug, error, trace, warn};

// Import tokens and lexer state from the sibling lexer module.
use super::lexer::AnsiToken;
// Import command enums from the sibling commands module.
use super::commands::{AnsiCommand, C0Control, CsiCommand};
// Import public constants from the lexer module that are used directly.
// Also import placeholder constants for specific ESC sequences.
use super::lexer::{APC, CSI, DCS, ESC, ESC_7, ESC_8, ESC_C, HTS, IND, NEL, OSC, PM, RI, SS2, SS3, ST};

/// Represents the state of the ANSI parser.
#[derive(Debug, PartialEq, Clone, Copy)]
enum ParserState {
    /// Initial state, ready to process tokens.
    Ground,
    /// Processing a CSI sequence.
    Csi,
    /// Processing a DCS sequence data string.
    Dcs,
    /// Processing an OSC string.
    Osc,
    /// Processing a PM string.
    Pm,
    /// Processing an APC string.
    Apc,
}

/// A parser that takes a stream of `AnsiToken`s and produces `AnsiCommand`s.
///
/// This parser implements a state machine to handle various ANSI escape sequences,
/// including CSI, OSC, DCS, PM, and APC sequences, as well as C0/C1 control codes
/// and printable characters.
pub struct AnsiParser {
    /// The current state of the parser state machine.
    state: ParserState,
    /// A buffer to store the commands generated during processing.
    commands: Vec<AnsiCommand>,
    /// Buffer for collecting numeric parameters within a CSI sequence.
    csi_params: Vec<usize>,
    /// Buffer for collecting intermediate bytes (0x20-0x2F) within a CSI sequence.
    csi_intermediates: Vec<u8>,
    /// Buffer for collecting data bytes within a DCS sequence.
    dcs_bytes: Vec<u8>,
    /// Buffer for collecting data bytes within an OSC sequence.
    osc_string_bytes: Vec<u8>,
    /// Buffer for collecting data bytes within a PM sequence.
    pm_string_bytes: Vec<u8>,
    /// Buffer for collecting data bytes within an APC sequence.
    apc_string_bytes: Vec<u8>,
}

impl AnsiParser {
    /// Creates a new `AnsiParser` in the `Ground` state.
    ///
    /// Initializes all internal buffers and sets the initial state.
    pub fn new() -> Self {
        AnsiParser {
            state: ParserState::Ground,
            commands: Vec::new(),
            csi_params: Vec::new(),
            csi_intermediates: Vec::new(),
            dcs_bytes: Vec::new(),
            osc_string_bytes: Vec::new(),
            pm_string_bytes: Vec::new(),
            apc_string_bytes: Vec::new(),
        }
    }

    /// Processes a single token and updates the parser's state and command list.
    ///
    /// This method drives the state machine based on the incoming token.
    /// Completed commands are added to the internal `commands` buffer.
    ///
    /// # Arguments
    ///
    /// * `token`: The `AnsiToken` received from the lexer.
    pub fn process_token(&mut self, token: AnsiToken) {
        trace!(
            "Parser Processing token: {:?} in state: {:?}",
            token, self.state
        );
        let initial_state = self.state;
        match initial_state {
            ParserState::Ground => self.handle_ground(token),
            ParserState::Csi => self.handle_csi(token),
            ParserState::Dcs => self.handle_dcs(token),
            ParserState::Osc => self.handle_osc(token),
            ParserState::Pm => self.handle_pm(token),
            ParserState::Apc => self.handle_apc(token),
        }
        // Log state transitions for debugging.
        if self.state != initial_state {
            trace!(
                "Parser State transition: {:?} -> {:?}",
                initial_state, self.state
            );
        }
    }

    /// Returns the list of commands processed so far and clears the internal buffer.
    ///
    /// This should be called periodically by the consumer to retrieve parsed commands.
    pub fn take_commands(&mut self) -> Vec<AnsiCommand> {
        std::mem::take(&mut self.commands)
    }

    /// Resets the parser state to `Ground` and clears all internal buffers.
    ///
    /// This is typically called when a sequence is completed or aborted.
    fn reset_state(&mut self) {
        self.state = ParserState::Ground;
        self.csi_params.clear();
        self.csi_intermediates.clear();
        self.dcs_bytes.clear();
        self.osc_string_bytes.clear();
        self.pm_string_bytes.clear();
        self.apc_string_bytes.clear();
    }

    /// Handles tokens received while the parser is in the `Ground` state.
    ///
    /// This state expects printable characters, C0 control codes, or sequence introducers (C1 or ESC+Fe).
    fn handle_ground(&mut self, token: AnsiToken) {
        match token {
            AnsiToken::Print(c) => self.commands.push(AnsiCommand::Print(c)),
            AnsiToken::C0Control(byte) => {
                // Convert the byte to a C0Control enum variant if possible.
                if let Some(c0_control) = C0Control::from_byte(byte) {
                    // ESC should not be received as a C0Control in Ground state if the lexer
                    // correctly transitions to the Escape state. Treat it as an error if it occurs.
                    if c0_control == C0Control::ESC {
                        warn!("Received unexpected ESC as C0Control in Ground state (lexer should transition state)");
                        self.commands.push(AnsiCommand::Error(byte));
                    } else {
                        self.commands.push(AnsiCommand::C0Control(c0_control));
                    }
                } else {
                    // This path should ideally not be reached if the lexer only sends valid C0 bytes.
                    warn!("Received non-C0 byte {:02X} as C0Control token", byte);
                    self.commands.push(AnsiCommand::Error(byte));
                }
            }
            AnsiToken::C1Control(byte) => {
                // Handle C1 control codes (0x80-0x9F) or placeholders for specific ESC sequences
                // generated by the lexer (e.g., ESC 7 -> ESC_7).
                match byte {
                    // Map common C1/ESC sequences to equivalent CSI commands for simpler handling downstream.
                    IND => self.commands.push(AnsiCommand::Csi(CsiCommand::CursorNextLine(1))),
                    NEL => self.commands.push(AnsiCommand::Csi(CsiCommand::CursorNextLine(1))), // NEL often treated like IND
                    HTS => self.commands.push(AnsiCommand::Csi(CsiCommand::SetTabStop)),
                    RI => self.commands.push(AnsiCommand::Csi(CsiCommand::CursorPrevLine(1))),
                    // Map SS2/SS3 to SCS commands (though full SCS handling might be more complex).
                    SS2 => self.commands.push(AnsiCommand::Csi(CsiCommand::SelectCharacterSet('(', 'N'))), // SS2 (ESC N) -> SCS G2 'N' (not standard, but placeholder)
                    SS3 => self.commands.push(AnsiCommand::Csi(CsiCommand::SelectCharacterSet(')', 'O'))), // SS3 (ESC O) -> SCS G3 'O' (not standard, but placeholder)
                    // Handle placeholder C1 codes mapped by the lexer for specific ESC sequences.
                    ESC_7 => self.commands.push(AnsiCommand::Csi(CsiCommand::SaveCursorAnsi)), // Mapped from ESC 7 (DECSC)
                    ESC_8 => self.commands.push(AnsiCommand::Csi(CsiCommand::RestoreCursorAnsi)), // Mapped from ESC 8 (DECRC)
                    ESC_C => self.commands.push(AnsiCommand::Csi(CsiCommand::Reset)), // Mapped from ESC c (RIS)
                    // Handle actual C1 introducer bytes by transitioning state.
                    CSI => {
                        self.push_token_and_change_state(AnsiToken::CsiEntry, ParserState::Csi);
                    }
                    DCS => {
                        self.push_token_and_change_state(AnsiToken::DcsEntry, ParserState::Dcs);
                    }
                    OSC => {
                        self.push_token_and_change_state(AnsiToken::OscString, ParserState::Osc);
                    }
                    PM => {
                        self.push_token_and_change_state(AnsiToken::PmString, ParserState::Pm);
                    }
                    APC => {
                        self.push_token_and_change_state(AnsiToken::ApcString, ParserState::Apc);
                    }
                    ST => {
                        // ST (String Terminator) in Ground state is usually ignored or treated as an error.
                        warn!("Received unexpected StringTerminator (ST) in Ground state.");
                        self.commands.push(AnsiCommand::Ignore(ST));
                    }
                    // Handle other C1 controls if needed, otherwise treat as unsupported/ignore.
                    _ => {
                        warn!("Unsupported C1 control byte: {:02X}", byte);
                        self.commands.push(AnsiCommand::C1Control(byte)); // Pass the raw byte
                    }
                }
            }
            // Handle entry tokens that signal the start of a sequence (usually emitted by lexer after C1 or ESC+Fe).
            AnsiToken::CsiEntry => {
                self.state = ParserState::Csi;
                self.csi_params.clear();
                self.csi_intermediates.clear();
            }
            AnsiToken::DcsEntry => {
                self.state = ParserState::Dcs;
                self.dcs_bytes.clear();
            }
            AnsiToken::OscString => {
                self.state = ParserState::Osc;
                self.osc_string_bytes.clear();
            }
            AnsiToken::PmString => {
                self.state = ParserState::Pm;
                self.pm_string_bytes.clear();
            }
            AnsiToken::ApcString => {
                self.state = ParserState::Apc;
                self.apc_string_bytes.clear();
            }
            AnsiToken::StringTerminator => {
                // ST in Ground state is usually ignored or treated as an error.
                warn!("Received unexpected StringTerminator token in Ground state.");
                self.commands.push(AnsiCommand::Ignore(ST));
            }
            AnsiToken::Ignored(byte) => {
                self.commands.push(AnsiCommand::Ignore(byte));
            }
            AnsiToken::Error(byte) => {
                error!("Lexer reported error byte: {:02X}", byte);
                self.commands.push(AnsiCommand::Error(byte));
            }
            // These tokens should not be received directly in Ground state if the lexer is correct.
            // Handle them defensively by emitting an error and staying in Ground state.
            AnsiToken::TwoByteEscape(_) | // Should be followed by C1Control
            AnsiToken::CsiParam(_) |
            AnsiToken::CsiIntermediate(_) |
            AnsiToken::CsiFinal(_) |
            AnsiToken::DcsParam(_) |
            AnsiToken::DcsIntermediate(_) |
            AnsiToken::DcsStringByte(_) |
            AnsiToken::OscStringByte(_) |
            AnsiToken::PmStringByte(_) |
            AnsiToken::ApcStringByte(_) => {
                warn!("Received unexpected token {:?} in Ground state.", token);
                if let Some(byte) = token_to_byte(&token) {
                    self.commands.push(AnsiCommand::Error(byte));
                } else {
                    // Use a generic error code if the token doesn't map to a single byte.
                    self.commands.push(AnsiCommand::Error(0xFF));
                }
                // Stay in Ground state
            }
        }
    }

    /// Helper to transition state and clear appropriate buffers when an entry token is received.
    ///
    /// This is used for C1 introducers (CSI, DCS, OSC, etc.) or equivalent ESC sequences
    /// that lead directly into a specific parsing state.
    fn push_token_and_change_state(&mut self, token: AnsiToken, next_state: ParserState) {
        // We don't actually push the *entry* token itself (e.g., CsiEntry) to the command list.
        // The entry token just signals the state transition. The parser will then process
        // subsequent tokens specific to that state (e.g., CsiParam, CsiIntermediate, CsiFinal).
        trace!(
            "Received entry token {:?}, changing state to {:?}",
            token, next_state
        );
        self.state = next_state;
        // Clear relevant buffers for the new state.
        match next_state {
            ParserState::Csi => {
                self.csi_params.clear();
                self.csi_intermediates.clear();
            }
            ParserState::Dcs => self.dcs_bytes.clear(),
            ParserState::Osc => self.osc_string_bytes.clear(),
            ParserState::Pm => self.pm_string_bytes.clear(),
            ParserState::Apc => self.apc_string_bytes.clear(),
            ParserState::Ground => { /* No buffers specific to Ground */ }
        }
    }


    /// Handles tokens received while the parser is in the `Csi` state.
    ///
    /// This state expects parameter bytes, intermediate bytes, or a final byte.
    /// C0 control codes (including ESC) abort the sequence.
    fn handle_csi(&mut self, token: AnsiToken) {
        match token {
            AnsiToken::C0Control(byte) => {
                // C0 controls inside CSI abort the sequence and are processed as C0 commands.
                if let Some(c0_control) = C0Control::from_byte(byte) {
                    // Special case: ESC within CSI aborts and emits C0Control(ESC).
                    if c0_control == C0Control::ESC {
                        warn!("CSI sequence aborted by ESC");
                        self.commands.push(AnsiCommand::C0Control(C0Control::ESC));
                    } else {
                        warn!("CSI sequence aborted by C0 control: 0x{:02X}", byte);
                        self.commands.push(AnsiCommand::C0Control(c0_control));
                    }
                } else {
                    // This path should not be reached if the lexer is correct.
                    warn!(
                        "Received non-C0 byte {:02X} as C0Control token in CSI state",
                        byte
                    );
                    self.commands.push(AnsiCommand::Error(byte));
                }
                self.reset_state(); // Abort CSI sequence.
            }
            AnsiToken::CsiParam(byte) => {
                // Parameter bytes (0-9, ;, <, =, >, ?)
                match byte {
                    b'0'..=b'9' => {
                        let digit = (byte - b'0') as usize;
                        // Ensure a parameter slot exists before trying to modify it.
                        if self.csi_params.is_empty() {
                            self.csi_params.push(0);
                        }
                        // Add the digit to the last parameter, handling potential overflow.
                        if let Some(last) = self.csi_params.last_mut() {
                            *last = last.saturating_mul(10).saturating_add(digit);
                        }
                    }
                    b';' => {
                        // Parameter separator. Add a new parameter slot (initialized to 0).
                        // Handle leading/multiple semicolons correctly.
                        if self.csi_params.is_empty() {
                            // Handle leading semicolon case: CSI ; -> params [0, 0]
                            self.csi_params.push(0);
                        }
                        self.csi_params.push(0); // Add new parameter slot.
                    }
                    b'<' | b'=' | b'>' | b'?' => {
                        // Private mode introducer or sub-parameter marker. Store as intermediate.
                        // Note: Standard CSI doesn't allow private markers after params start,
                        // but some terminals might. We store it as an intermediate.
                        if self.csi_intermediates.is_empty() {
                            self.csi_intermediates.push(byte);
                        } else {
                            // Multiple private markers/intermediates can be complex.
                            // For simplicity, store only the first for now.
                            warn!(
                                "Received multiple private marker/intermediate bytes in CSI: {:02X}, storing first.",
                                byte
                            );
                            // Optionally push an error or ignore subsequent markers.
                            // self.commands.push(AnsiCommand::Error(byte));
                        }
                    }
                    _ => {
                        // Includes ':' (sub-parameters, not fully handled yet).
                        warn!("Unsupported CSI parameter byte: {:02X}", byte);
                        self.commands.push(AnsiCommand::Error(byte));
                        // Don't necessarily abort here; maybe the final byte is valid?
                        // Stay in Csi state for now.
                    }
                }
            }
            AnsiToken::CsiIntermediate(byte) => {
                // Intermediate bytes (0x20-0x2F).
                // Limit the number of intermediates stored (typically max 2).
                const MAX_CSI_INTERMEDIATES: usize = 2;
                if self.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                    self.csi_intermediates.push(byte);
                } else {
                    warn!("Too many CSI intermediate bytes, ignoring: {:02X}", byte);
                    self.commands.push(AnsiCommand::Error(byte));
                    // Don't change state, wait for final byte.
                }
            }
            AnsiToken::CsiFinal(byte) => {
                // Final byte (0x40-0x7E) - marks the end of the CSI sequence.
                self.parse_csi_command(byte);
                self.reset_state(); // CSI sequence finished, return to Ground state.
            }
            // C1 controls also abort CSI.
            AnsiToken::C1Control(byte) => {
                warn!("CSI sequence aborted by C1 control: 0x{:02X}", byte);
                self.commands.push(AnsiCommand::C1Control(byte));
                self.reset_state();
            }
            AnsiToken::Error(byte) => {
                // Handle Error(ESC) specifically as an abort, emitting C0Control::ESC.
                if byte == ESC {
                    warn!("CSI sequence aborted by ESC (via Error token)");
                    self.commands.push(AnsiCommand::C0Control(C0Control::ESC));
                } else {
                    error!("Lexer reported error byte: {:02X} in CSI state", byte);
                    self.commands.push(AnsiCommand::Error(byte));
                }
                self.reset_state(); // Abort CSI sequence on error.
            }
            // Other tokens are unexpected in the CSI state.
            _ => {
                warn!(
                    "Received unexpected token {:?} in Csi state, aborting sequence.",
                    token
                );
                if let Some(byte) = token_to_byte(&token) {
                    self.commands.push(AnsiCommand::Error(byte));
                } else {
                    self.commands.push(AnsiCommand::Error(0xFF)); // Generic error code
                }
                self.reset_state(); // Abort CSI sequence.
            }
        }
    }


    /// Parses the collected CSI parameters and final byte into a specific `CsiCommand`.
    ///
    /// This function interprets the meaning of the CSI sequence based on its final byte
    /// and any collected parameters or intermediate bytes.
    fn parse_csi_command(&mut self, final_byte: u8) {
        // Helper to get parameter with default value.
        // Crucially handles the case where csi_params might be empty or have fewer params than requested.
        // It also implements the common terminal behavior where a parameter value of 0 often
        // defaults to 1 (e.g., for cursor movement), unless 0 has a specific meaning (e.g., for ED/EL).
        let get_param = |parser: &AnsiParser, index: usize, default: usize| -> usize {
            match parser.csi_params.get(index).copied() {
                Some(0) => default, // Treat explicit 0 as default (usually 1)
                Some(val) => val,   // Use the provided value
                None => default,    // Use default if parameter is missing
            }
        };
        // Helper to get parameter defaulting to 0, used where 0 is a distinct valid value.
        let get_param_or_0 = |parser: &AnsiParser, index: usize| -> usize {
             parser.csi_params.get(index).copied().unwrap_or(0)
        };


        // Get parameters with appropriate defaults for common cases.
        let p1 = get_param(self, 0, 1); // Default 1 for most commands (CUU, CUD, IL, DL, etc.)
        let p0 = get_param_or_0(self, 0); // Default 0 for ED, EL, DSR, TBC etc.


        trace!(
            "Dispatch CSI: final='{}', params={:?}, intermediates={:?}",
            final_byte as char, self.csi_params, self.csi_intermediates
        );

        // Handle private sequences based on intermediate bytes (primarily '?').
        // Note: Other intermediates like '>', '=', '!' exist but are less common or vendor-specific.
        let is_private = self.csi_intermediates.first() == Some(&b'?');
        let introducer = self.csi_intermediates.first().copied(); // Store the first intermediate/private marker

        let command = match (introducer, final_byte) {
            // --- Standard Sequences (No Intermediate or handled private marker like '?') ---
            // Note: We often allow '?' for common sequences where terminals might accept it,
            // even if not strictly standard (e.g., CSI ? 6 n for DSR).
            (None | Some(b'?'), b'A') => CsiCommand::CursorUp(p1),
            (None | Some(b'?'), b'B') => CsiCommand::CursorDown(p1),
            (None | Some(b'?'), b'C') => CsiCommand::CursorForward(p1),
            (None | Some(b'?'), b'D') => CsiCommand::CursorBackward(p1),
            (None | Some(b'?'), b'E') => CsiCommand::CursorNextLine(p1),
            (None | Some(b'?'), b'F') => CsiCommand::CursorPrevLine(p1),
            (None | Some(b'?'), b'G') => CsiCommand::CursorCharacterAbsolute(p1),
            (None | Some(b'?'), b'H') | (None | Some(b'?'), b'f') => { // CUP / HVP
                // Default to 1,1 if params are missing or 0.
                let row = get_param(self, 0, 1);
                let col = get_param(self, 1, 1);
                CsiCommand::CursorPosition(row, col)
            }
            (None | Some(b'?'), b'J') => CsiCommand::EraseInDisplay(p0), // Use p0 (default 0)
            (None | Some(b'?'), b'K') => CsiCommand::EraseInLine(p0),    // Use p0 (default 0)
            (None | Some(b'?'), b'L') => CsiCommand::InsertLine(p1),
            (None | Some(b'?'), b'M') => CsiCommand::DeleteLine(p1),
            (None | Some(b'?'), b'P') => CsiCommand::DeleteCharacter(p1),
            (None | Some(b'?'), b'X') => CsiCommand::EraseCharacter(p1),
            (None | Some(b'?'), b'@') => CsiCommand::InsertCharacter(p1),
            (None | Some(b'?'), b'm') => CsiCommand::SetGraphicsRendition(self.csi_params.clone()), // Pass all params for SGR
            (None | Some(b'?'), b's') => CsiCommand::SaveCursor, // SCO Save Cursor (ANSI.SYS compatible)
            (None | Some(b'?'), b'u') => CsiCommand::RestoreCursor, // SCO Restore Cursor (ANSI.SYS compatible)
            (None | Some(b'?'), b'h') => { // SM / DECSET
                if is_private {
                    CsiCommand::SetModePrivate(self.csi_params.clone())
                } else {
                    CsiCommand::SetMode(self.csi_params.clone())
                }
            }
            (None | Some(b'?'), b'l') => { // RM / DECRST
                if is_private {
                    CsiCommand::ResetModePrivate(self.csi_params.clone())
                } else {
                    CsiCommand::ResetMode(self.csi_params.clone())
                }
            }
            (None | Some(b'?'), b'n') => CsiCommand::DeviceStatusReport(p0), // Use p0 (default 0)
            (None | Some(b'?'), b'c') => CsiCommand::IdentifyTerminal, // DA (Device Attributes)
            (None | Some(b'?'), b'g') => CsiCommand::ClearTabStops(p0), // TBC (Tabulation Clear) - Use p0 (default 0)
            (None | Some(b'?'), b'r') => { // DECSTBM / DECSLRM
                if is_private {
                    // DECSLRM ?r - Set Left/Right Margins (Less common)
                    warn!("DECSLRM (CSI ? r) not fully implemented.");
                    // Provide default values or handle specific parameters if needed.
                    let left = get_param(self, 0, 1);
                    let right = get_param(self, 1, 0); // 0 might mean terminal width
                     CsiCommand::Unsupported(vec![left, right], Some(final_byte)) // Treat as unsupported for now
                } else {
                    // DECSTBM - Set Top/Bottom Margins
                    let top = get_param(self, 0, 1);
                    // Default bottom is tricky, often screen height. Use 0 if only 1 param?
                    // Let's default to 0, indicating full height if only one param is given.
                    let bottom = if self.csi_params.len() >= 2 {
                        get_param(self, 1, 0) // Default to 0 if second param is 0 or missing
                    } else {
                        0 // Default to 0 (implies full height) if only one param
                    };
                    CsiCommand::ScrollingRegion(top, bottom) // Let Term logic handle 0/default bottom
                }
            }
            (None | Some(b'?'), b't') => { // Window manipulation (xterm extensions)
                warn!("Window Manipulation (CSI t) sequence received - Not Implemented: {:?}", self.csi_params);
                CsiCommand::Unsupported(self.csi_params.clone(), Some(final_byte))
            }
            (None | Some(b'?'), b'y') => { // DECRQPSR (Request Status Report) - Less common
                CsiCommand::RequestTerminalParameters(p0) // Use p0 (default 0)
            }
            (None | Some(b'?'), b'q') => { // DECSCUSR (Set Cursor Style)
                 // Note: Space intermediate is more standard (CSI Ps SP q), but handle without too.
                 CsiCommand::SetCursorStyle(p0) // Use p0 (default 0)
            }
             // --- Sequences with specific intermediates ---
             (Some(b' '), b'q') => CsiCommand::SetCursorStyle(p0), // Standard DECSCUSR with space intermediate

            // --- Catch-all for unsupported sequences ---
            _ => {
                warn!(
                    "Unsupported CSI sequence: ESC [ {:?} {:?} {}",
                    self.csi_intermediates.iter().map(|&b| b as char).collect::<String>(),
                    self.csi_params,
                    final_byte as char
                );
                CsiCommand::Unsupported(self.csi_params.clone(), Some(final_byte))
            }
        };

        self.commands.push(AnsiCommand::Csi(command));
    }


    /// Handles tokens in the `Dcs` state (collecting the data string).
    ///
    /// Expects data bytes or a String Terminator (ST). C0 controls (except ESC) are ignored.
    /// ESC aborts the sequence.
    fn handle_dcs(&mut self, token: AnsiToken) {
        match token {
            AnsiToken::DcsStringByte(byte) => {
                // Collect data bytes for the DCS string.
                self.dcs_bytes.push(byte);
            }
            AnsiToken::StringTerminator => {
                // DCS sequence finished with ST. Emit the command and reset state.
                self.commands.push(AnsiCommand::Dcs(std::mem::take(&mut self.dcs_bytes)));
                self.commands.push(AnsiCommand::StringTerminator); // Push terminator command itself
                self.reset_state();
            }
            AnsiToken::C0Control(byte) => {
                if let Some(C0Control::ESC) = C0Control::from_byte(byte) {
                    // Abort sequence on ESC, push ESC command, and reset.
                    warn!("DCS sequence aborted by ESC");
                    self.commands.push(AnsiCommand::C0Control(C0Control::ESC));
                    self.reset_state();
                } else {
                    // Ignore other C0 controls within the DCS string.
                    trace!("Ignoring C0 control 0x{:02X} in DCS state", byte);
                    self.commands.push(AnsiCommand::Ignore(byte));
                }
            }
            AnsiToken::Error(byte) => {
                // Handle Error(ESC) specifically as an abort.
                if byte == ESC {
                    warn!("DCS sequence aborted by ESC (via Error token)");
                    self.commands.push(AnsiCommand::C0Control(C0Control::ESC));
                } else {
                    error!("Lexer reported error byte: {:02X} in DCS state", byte);
                    self.commands.push(AnsiCommand::Error(byte));
                }
                self.reset_state(); // Abort DCS sequence on error.
            }
            // Any other token is unexpected in this state.
            _ => {
                warn!(
                    "Unexpected token {:?} in Dcs state, aborting sequence.",
                    token
                );
                if let Some(b) = token_to_byte(&token) {
                    self.commands.push(AnsiCommand::Error(b));
                } else {
                    self.commands.push(AnsiCommand::Error(0xFF));
                }
                self.reset_state();
            }
        }
    }

    /// Handles tokens in the `Osc` state (collecting the string).
    ///
    /// Expects data bytes or a terminator (BEL or ST). C0 controls (except BEL, ESC) are ignored.
    /// ESC aborts the sequence.
    fn handle_osc(&mut self, token: AnsiToken) {
        match token {
            AnsiToken::OscStringByte(byte) => {
                // Collect data bytes for the OSC string.
                self.osc_string_bytes.push(byte);
            }
            AnsiToken::StringTerminator => {
                // OSC sequence finished with ST. Emit the command and reset state.
                self.commands.push(AnsiCommand::Osc(std::mem::take(&mut self.osc_string_bytes)));
                self.commands.push(AnsiCommand::StringTerminator); // Push terminator command itself
                self.reset_state();
            }
            AnsiToken::C0Control(byte) => {
                if let Some(C0Control::BEL) = C0Control::from_byte(byte) {
                    // BEL also terminates OSC sequences.
                    self.commands.push(AnsiCommand::Osc(std::mem::take(&mut self.osc_string_bytes)));
                    self.commands.push(AnsiCommand::C0Control(C0Control::BEL)); // Push BEL command itself
                    self.reset_state();
                } else if let Some(C0Control::ESC) = C0Control::from_byte(byte) {
                    // Abort sequence on ESC, push ESC command, and reset.
                    warn!("OSC sequence aborted by ESC");
                    self.commands.push(AnsiCommand::C0Control(C0Control::ESC));
                    self.reset_state();
                } else {
                    // Ignore other C0 controls within the OSC string.
                    trace!("Ignoring C0 control 0x{:02X} in OSC state", byte);
                    self.commands.push(AnsiCommand::Ignore(byte));
                }
            }
            AnsiToken::Error(byte) => {
                // Handle Error(ESC) specifically as an abort.
                if byte == ESC {
                    warn!("OSC sequence aborted by ESC (via Error token)");
                    self.commands.push(AnsiCommand::C0Control(C0Control::ESC));
                } else {
                    error!("Lexer reported error byte: {:02X} in OSC state", byte);
                    self.commands.push(AnsiCommand::Error(byte));
                }
                self.reset_state(); // Abort OSC sequence on error.
            }
            // Any other token is unexpected in this state.
            _ => {
                warn!(
                    "Unexpected token {:?} in Osc state, aborting sequence.",
                    token
                );
                if let Some(b) = token_to_byte(&token) {
                    self.commands.push(AnsiCommand::Error(b));
                } else {
                    self.commands.push(AnsiCommand::Error(0xFF));
                }
                self.reset_state();
            }
        }
    }

    /// Handles tokens in the `Pm` state (collecting the string).
    ///
    /// Expects data bytes or a String Terminator (ST). C0 controls (except ESC) are ignored.
    /// ESC aborts the sequence.
    fn handle_pm(&mut self, token: AnsiToken) {
        match token {
            AnsiToken::PmStringByte(byte) => {
                // Collect data bytes for the PM string.
                self.pm_string_bytes.push(byte);
            }
            AnsiToken::StringTerminator => {
                // PM sequence finished with ST. Emit the command and reset state.
                self.commands.push(AnsiCommand::Pm(std::mem::take(&mut self.pm_string_bytes)));
                self.commands.push(AnsiCommand::StringTerminator); // Push terminator command itself
                self.reset_state();
            }
            AnsiToken::C0Control(byte) => {
                if let Some(C0Control::ESC) = C0Control::from_byte(byte) {
                    // Abort sequence on ESC, push ESC command, and reset.
                    warn!("PM sequence aborted by ESC");
                    self.commands.push(AnsiCommand::C0Control(C0Control::ESC));
                    self.reset_state();
                } else {
                    // Ignore other C0 controls within the PM string.
                    trace!("Ignoring C0 control 0x{:02X} in PM state", byte);
                    self.commands.push(AnsiCommand::Ignore(byte));
                }
            }
            AnsiToken::Error(byte) => {
                // Handle Error(ESC) specifically as an abort.
                if byte == ESC {
                    warn!("PM sequence aborted by ESC (via Error token)");
                    self.commands.push(AnsiCommand::C0Control(C0Control::ESC));
                } else {
                    error!("Lexer reported error byte: {:02X} in PM state", byte);
                    self.commands.push(AnsiCommand::Error(byte));
                }
                self.reset_state(); // Abort PM sequence on error.
            }
            // Any other token is unexpected in this state.
            _ => {
                warn!(
                    "Unexpected token {:?} in Pm state, aborting sequence.",
                    token
                );
                if let Some(b) = token_to_byte(&token) {
                    self.commands.push(AnsiCommand::Error(b));
                } else {
                    self.commands.push(AnsiCommand::Error(0xFF));
                }
                self.reset_state();
            }
        }
    }

    /// Handles tokens in the `Apc` state (collecting the string).
    ///
    /// Expects data bytes or a String Terminator (ST). C0 controls (except ESC) are ignored.
    /// ESC aborts the sequence.
    fn handle_apc(&mut self, token: AnsiToken) {
        match token {
            AnsiToken::ApcStringByte(byte) => {
                // Collect data bytes for the APC string.
                self.apc_string_bytes.push(byte);
            }
            AnsiToken::StringTerminator => {
                // APC sequence finished with ST. Emit the command and reset state.
                self.commands.push(AnsiCommand::Apc(std::mem::take(&mut self.apc_string_bytes)));
                self.commands.push(AnsiCommand::StringTerminator); // Push terminator command itself
                self.reset_state();
            }
            AnsiToken::C0Control(byte) => {
                if let Some(C0Control::ESC) = C0Control::from_byte(byte) {
                    // Abort sequence on ESC, push ESC command, and reset.
                    warn!("APC sequence aborted by ESC");
                    self.commands.push(AnsiCommand::C0Control(C0Control::ESC));
                    self.reset_state();
                } else {
                    // Ignore other C0 controls within the APC string.
                    trace!("Ignoring C0 control 0x{:02X} in APC state", byte);
                    self.commands.push(AnsiCommand::Ignore(byte));
                }
            }
            AnsiToken::Error(byte) => {
                // Handle Error(ESC) specifically as an abort.
                if byte == ESC {
                    warn!("APC sequence aborted by ESC (via Error token)");
                    self.commands.push(AnsiCommand::C0Control(C0Control::ESC));
                } else {
                    error!("Lexer reported error byte: {:02X} in APC state", byte);
                    self.commands.push(AnsiCommand::Error(byte));
                }
                self.reset_state(); // Abort APC sequence on error.
            }
            // Any other token is unexpected in this state.
            _ => {
                warn!(
                    "Unexpected token {:?} in Apc state, aborting sequence.",
                    token
                );
                if let Some(b) = token_to_byte(&token) {
                    self.commands.push(AnsiCommand::Error(b));
                } else {
                    self.commands.push(AnsiCommand::Error(0xFF));
                }
                self.reset_state();
            }
        }
    }
}

/// Helper function to attempt to get a byte value from a token for error reporting.
///
/// Returns `Some(byte)` if the token represents a single byte, `None` otherwise.
fn token_to_byte(token: &AnsiToken) -> Option<u8> {
    match token {
        AnsiToken::Print(c) => {
            // Only return a byte if the char is single-byte ASCII.
            let mut bytes = [0; 4];
            let len = c.encode_utf8(&mut bytes).len();
            if len == 1 { Some(bytes[0]) } else { None }
        }
        AnsiToken::C0Control(byte) => Some(*byte),
        AnsiToken::C1Control(byte) => Some(*byte),
        AnsiToken::TwoByteEscape(byte) => Some(*byte),
        AnsiToken::CsiParam(byte) => Some(*byte),
        AnsiToken::CsiIntermediate(byte) => Some(*byte),
        AnsiToken::CsiFinal(byte) => Some(*byte),
        AnsiToken::DcsParam(byte) => Some(*byte),
        AnsiToken::DcsIntermediate(byte) => Some(*byte),
        AnsiToken::DcsStringByte(byte) => Some(*byte),
        AnsiToken::OscStringByte(byte) => Some(*byte),
        AnsiToken::PmStringByte(byte) => Some(*byte),
        AnsiToken::ApcStringByte(byte) => Some(*byte),
        AnsiToken::StringTerminator => Some(ST), // Use ST constant as representative byte.
        AnsiToken::Ignored(byte) => Some(*byte),
        AnsiToken::Error(byte) => Some(*byte),
        // For tokens without a direct single-byte representation, return None.
        AnsiToken::CsiEntry
        | AnsiToken::DcsEntry
        | AnsiToken::OscString
        | AnsiToken::PmString
        | AnsiToken::ApcString => None,
    }
}
