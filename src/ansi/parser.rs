// src/ansi/parser.rs

//! Parses input byte streams, including escape sequences (CSI, OSC),
//! and outputs structured AnsiCommand enums.
//!
//! This code is adapted from the original myterm/src/term/parser.rs,
//! refactoring the state machine to produce command objects instead of
//! directly modifying terminal state.

use super::commands::{AnsiCommand, C0Control, CsiCommand, OscCommand, SgrParameter, EraseMode, DecPrivateMode};
use super::{MAX_CSI_PARAMS, MAX_CSI_INTERMEDIATES, MAX_OSC_STRING_LEN};
use std::collections::VecDeque; // Using VecDeque to buffer output commands
use log::{trace, debug, warn, error};
use utf8parse::{Parser as Utf8Parser, Receiver as Utf8Receiver}; // For handling UTF-8 characters

// --- Constants for Control Codes ---
#[allow(dead_code)] const NUL: u8 = 0x00;
#[allow(dead_code)] const SOH: u8 = 0x01;
#[allow(dead_code)] const STX: u8 = 0x02;
#[allow(dead_code)] const ETX: u8 = 0x03;
#[allow(dead_code)] const EOT: u8 = 0x04;
#[allow(dead_code)] const ENQ: u8 = 0x05;
#[allow(dead_code)] const ACK: u8 = 0x06;
const BEL: u8 = 0x07;
const BS: u8 = 0x08;
const HT: u8 = 0x09;
const LF: u8 = 0x0A;
const VT: u8 = 0x0B;
const FF: u8 = 0x0C;
const CR: u8 = 0x0D;
const SO: u8 = 0x0E;
const SI: u8 = 0x0F;
#[allow(dead_code)] const DLE: u8 = 0x10;
#[allow(dead_code)] const DC1: u8 = 0x11; // XON
#[allow(dead_code)] const DC2: u8 = 0x12;
#[allow(dead_code)] const DC3: u8 = 0x13; // XOFF
#[allow(dead_code)] const DC4: u8 = 0x14;
#[allow(dead_code)] const NAK: u8 = 0x15;
#[allow(dead_code)] const SYN: u8 = 0x16;
#[allow(dead_code)] const ETB: u8 = 0x17;
#[allow(dead_code)] const CAN: u8 = 0x18;
#[allow(dead_code)] const EM: u8 = 0x19;
const SUB: u8 = 0x1A;
pub(super) const ESC: u8 = 0x1B; // Made pub(super)
#[allow(dead_code)] const FS: u8 = 0x1C;
#[allow(dead_code)] const GS: u8 = 0x1D;
#[allow(dead_code)] const RS: u8 = 0x1E;
#[allow(dead_code)] const US: u8 = 0x1F;
pub(super) const DEL: u8 = 0x7F;

// C1 Controls
const IND: u8 = 0x84;
const NEL: u8 = 0x85;
const HTS: u8 = 0x88;
const RI: u8 = 0x8D;
#[allow(dead_code)] const SS2: u8 = 0x8E;
#[allow(dead_code)] const SS3: u8 = 0x8F;
#[allow(dead_code)] const DCS: u8 = 0x90;
const CSI: u8 = 0x9B;
const ST: u8 = 0x9C;
const OSC: u8 = 0x9D;
#[allow(dead_code)] const PM: u8 = 0x9E;
#[allow(dead_code)] const APC: u8 = 0x9F;

// --- Constants for SGR ---
const SGR_RESET: u16 = 0;
const SGR_BOLD: u16 = 1;
const SGR_FAINT: u16 = 2;
const SGR_ITALIC: u16 = 3;
const SGR_UNDERLINE: u16 = 4;
const SGR_BLINK_SLOW: u16 = 5;
const SGR_BLINK_FAST: u16 = 6;
const SGR_REVERSE: u16 = 7;
const SGR_HIDDEN: u16 = 8;
const SGR_STRIKETHROUGH: u16 = 9;
const SGR_NORMAL_INTENSITY: u16 = 22;
const SGR_ITALIC_OFF: u16 = 23;
const SGR_UNDERLINE_OFF: u16 = 24;
const SGR_BLINK_OFF: u16 = 25;
const SGR_REVERSE_OFF: u16 = 27;
const SGR_HIDDEN_OFF: u16 = 28;
const SGR_STRIKETHROUGH_OFF: u16 = 29;
// Basic color ranges (30-37 foreground, 40-47 background)
const SGR_FG_OFFSET: u16 = 30;
const SGR_FG_END: u16 = 37; // End of basic foreground range
const SGR_BG_OFFSET: u16 = 40;
const SGR_BG_END: u16 = 47; // End of basic background range
// Bright color ranges (90-97 foreground, 100-107 background)
const SGR_FG_BRIGHT_OFFSET: u16 = 90;
const SGR_FG_BRIGHT_END: u16 = 97; // End of bright foreground range
const SGR_BG_BRIGHT_OFFSET: u16 = 100;
const SGR_BG_BRIGHT_END: u16 = 107; // End of bright background range
// Offset to map SGR bright color codes to internal palette indices (8-15)
const SGR_BRIGHT_COLOR_OFFSET: u16 = 8;
// Default color codes
const SGR_FG_DEFAULT: u16 = 39;
const SGR_BG_DEFAULT: u16 = 49;
// Extended color codes
const SGR_EXTENDED_FG: u16 = 38;
const SGR_EXTENDED_BG: u16 = 48;
const SGR_EXTENDED_MODE_256: u16 = 5;
const SGR_EXTENDED_MODE_RGB: u16 = 2;

// --- Constants for DEC Private Modes (CSI ? Pn h/l) ---
const DECCKM: u16 = 1;
#[allow(dead_code)] const DECANM: u16 = 2;
#[allow(dead_code)] const DECCOLM: u16 = 3;
#[allow(dead_code)] const DECSCLM: u16 = 4;
#[allow(dead_code)] const DECSCNM: u16 = 5;
const DECOM: u16 = 6;
#[allow(dead_code)] const DECAWM: u16 = 7;
#[allow(dead_code)] const DECARM: u16 = 8;
#[allow(dead_code)] const X10_MOUSE: u16 = 9;
const BLINK_CURSOR_ATT610: u16 = 12;
#[allow(dead_code)] const DECPFF: u16 = 18;
#[allow(dead_code)] const DECPEX: u16 = 19;
const DECTCEM: u16 = 25;
#[allow(dead_code)] const DECNRCM: u16 = 42;
const ALT_SCREEN_BUF_47: u16 = 47;
#[allow(dead_code)] const MOUSE_HIGHLIGHT: u16 = 1001;
#[allow(dead_code)] const MOUSE_UTF8: u16 = 1005;
#[allow(dead_code)] const MOUSE_URXVT: u16 = 1015;
#[allow(dead_code)] const MOUSE_BTN_REPORT: u16 = 1000;
#[allow(dead_code)] const MOUSE_BTN_MOTION_REPORT: u16 = 1002;
#[allow(dead_code)] const MOUSE_ANY_MOTION_REPORT: u16 = 1003;
#[allow(dead_code)] const FOCUS_EVENT_REPORT: u16 = 1004;
#[allow(dead_code)] const MOUSE_SGR_REPORT: u16 = 1006;
#[allow(dead_code)] const INPUT_MODE_8BIT: u16 = 1034;
const ALT_SCREEN_BUF_1047: u16 = 1047;
const CURSOR_SAVE_RESTORE_1048: u16 = 1048;
const ALT_SCREEN_SAVE_RESTORE_1049: u16 = 1049;
#[allow(dead_code)] const BRACKETED_PASTE: u16 = 2004;

/// States for the ANSI escape sequence parser state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ParserState {
    /// Default state: expecting printable characters or C0/ESC control codes.
    #[default]
    Ground,
    /// Received ESC (0x1B), expecting a subsequent byte to determine sequence type.
    Escape,
    /// Received CSI (ESC [ or C1 0x9B), expecting parameters, intermediates, or final byte.
    CSIEntry,
    /// Parsing CSI parameters (digits 0-9 and ';').
    CSIParam,
    /// Parsing CSI intermediate bytes (0x20-0x2F).
    CSIIntermediate,
    /// Ignoring remaining bytes of a CSI sequence until a final byte (e.g., due to too many params/intermediates).
    CSIIgnore,
    /// Parsing an OSC string, collecting bytes until a terminator (BEL or ST).
    OSCString,
    /// Parsing a DCS sequence. (Placeholder)
    DCS,
    // Add states for other sequence types (SOS, APC, PM, etc.) as needed.
}

/// The ANSI parser state machine.
pub struct AnsiParser {
    state: ParserState,
    csi_params: Vec<u16>,
    csi_intermediates: Vec<char>,
    osc_string: Vec<u8>, // Use Vec<u8> for raw bytes in OSC string
    // Add fields for other sequence types (DCS, etc.) if needed
    utf8_parser: Utf8Parser, // For handling multi-byte UTF-8 characters
    command_queue: VecDeque<AnsiCommand>, // Buffer for parsed commands
}

impl Default for AnsiParser {
    fn default() -> Self {
        Self::new()
    }
}

impl AnsiParser {
    /// Creates a new ANSI parser in the initial state.
    pub fn new() -> Self {
        AnsiParser {
            state: ParserState::Ground,
            csi_params: Vec::with_capacity(MAX_CSI_PARAMS),
            csi_intermediates: Vec::with_capacity(MAX_CSI_INTERMEDIATES),
            osc_string: Vec::with_capacity(MAX_OSC_STRING_LEN),
            utf8_parser: Utf8Parser::new(),
            command_queue: VecDeque::new(),
        }
    }

    /// Feeds a single byte into the parser state machine.
    /// Returns a Vec of parsed commands that are ready to be processed.
    pub fn feed(&mut self, byte: u8) -> Vec<AnsiCommand> {
        // Use the Utf8Parser to handle multi-byte characters first
        self.utf8_parser.advance(&mut Utf8CommandReceiver { parser: self }, byte);

        // The Utf8CommandReceiver pushes completed characters/commands to the queue.
        // Any non-UTF8 bytes (like controls or escape sequences) are handled
        // by the state machine logic below.

        // Process the byte based on the current state
        match self.state {
            ParserState::Ground => self.handle_ground(byte),
            ParserState::Escape => self.handle_escape(byte),
            ParserState::CSIEntry => self.handle_csi_entry(byte),
            ParserState::CSIParam => self.handle_csi_param(byte),
            ParserState::CSIIntermediate => self.handle_csi_intermediate(byte),
            ParserState::CSIIgnore => self.handle_csi_ignore(byte),
            ParserState::OSCString => self.handle_osc_string(byte),
            ParserState::DCS => self.handle_dcs(byte),
        }

        // Return any commands that were generated during this byte's processing
        self.drain_command_queue()
    }

    /// Feeds a slice of bytes into the parser.
    /// Returns a Vec of all parsed commands from the slice.
    pub fn feed_slice(&mut self, bytes: &[u8]) -> Vec<AnsiCommand> {
        for &byte in bytes {
            self.feed(byte); // Processing each byte individually
        }
        self.drain_command_queue()
    }

    /// Drains the internal command queue and returns its contents as a Vec.
    fn drain_command_queue(&mut self) -> Vec<AnsiCommand> {
        self.command_queue.drain(..).collect()
    }

    /// Pushes a command onto the internal queue.
    fn push_command(&mut self, command: AnsiCommand) {
        self.command_queue.push_back(command);
    }

    // --- State Transition Handlers (Adapted from original parser.rs) ---

    /// Handles bytes when in the Ground state.
    fn handle_ground(&mut self, byte: u8) {
        match byte {
            0x20..=0x7E => { // Printable ASCII characters
                // Utf8Parser handles multi-byte characters, but single-byte ASCII
                // might also pass through here if not part of a sequence.
                // The Utf8CommandReceiver should catch these too, but this is a fallback.
                // For simplicity and correctness with UTF-8, rely on Utf8Parser for Print.
                // Any byte reaching here that isn't a control or ESC is likely an error
                // or part of a sequence we don't handle yet.
                 if byte.is_ascii_graphic() || byte == b' ' {
                     // This case should ideally be handled by the Utf8Parser,
                     // but as a fallback, push if it's a printable ASCII.
                     // The Utf8Parser integration is the primary path for text.
                     trace!("Ground: Received printable ASCII byte 0x{:02X}, relying on Utf8Parser", byte);
                 } else {
                     trace!("Ground: Ignoring unexpected byte 0x{:02X}", byte);
                 }
            }
            NUL..=US | DEL => { // C0 Control Codes
                self.push_command(AnsiCommand::C0Control(C0Control::from_byte(byte)));
            }
            ESC => { // Escape
                trace!("Ground -> Escape");
                self.state = ParserState::Escape;
                self.clear_csi_params();
                self.clear_osc_string();
            }
            IND_C1 => self.push_command(AnsiCommand::Csi(CsiCommand::CursorNextLine(1))), // C1 IND
            NEL_C1 => self.push_command(AnsiCommand::Csi(CsiCommand::CursorNextLine(1))), // C1 NEL
            HTS_C1 => self.push_command(AnsiCommand::Csi(CsiCommand::SetTabStop)), // C1 HTS
            RI_C1 => self.push_command(AnsiCommand::Csi(CsiCommand::CursorPrevLine(1))), // C1 RI
            CSI_C1 => { // C1 CSI
                trace!("Ground -> CSIEntry (via C1 CSI)");
                self.clear_csi_params();
                self.state = ParserState::CSIEntry;
            }
            OSC_C1 => { // C1 OSC
                 trace!("Ground -> OSCString (via C1 OSC)");
                 self.clear_osc_string();
                 self.state = ParserState::OSCString;
            }
            ST_C1 => { trace!("Ignoring ST (0x9C) in Ground state"); } // C1 ST
            DCS_C1 => { trace!("Ground -> DCS (via C1 DCS)"); self.state = ParserState::DCS; } // C1 DCS
            PM_C1 => { trace!("Ground: Ignoring C1 PM (0x9E)"); } // C1 PM
            APC_C1 => { trace!("Ground: Ignoring C1 APC (0x9F)"); } // C1 APC

            // Ignore other C1 controls we don't explicitly map to commands yet
            0x80..=0x9A | 0x9E..=0x9F => {
                 trace!("Ground: Ignoring unhandled C1 control 0x{:02X}", byte);
                 self.push_command(AnsiCommand::Ignore(byte));
            }
            _ => {
                 // Non-ASCII printable characters are handled by Utf8Parser
                 // Anything else here might be unexpected
                 trace!("Ground: Received unexpected byte 0x{:02X}", byte);
                 self.push_command(AnsiCommand::Ignore(byte));
            }
        }
    }

    /// Handles bytes after an ESC (0x1B) has been received.
    fn handle_escape(&mut self, byte: u8) {
        match byte {
            b'[' => {
                trace!("Escape -> CSIEntry");
                self.state = ParserState::CSIEntry;
            }
            b']' => {
                trace!("Escape -> OSCString");
                self.state = ParserState::OSCString;
            }
            b'\\' => { // ST (ESC \)
                 trace!("Escape -> Ground (received ST)");
                 // An ST after ESC outside of a sequence is usually ignored or an error.
                 // If it terminates an OSC/DCS sequence, it's handled in that state.
                 // Here, it likely indicates an aborted sequence.
                 self.push_command(AnsiCommand::Ignore(byte)); // Or AnsiCommand::Error("Aborted sequence".into())
                 self.state = ParserState::Ground;
            }
            b'N' => { trace!("Escape -> Ground (SS2)"); self.push_command(AnsiCommand::EscapeSequence('N', vec![])); self.state = ParserState::Ground; } // SS2
            b'O' => { trace!("Escape -> Ground (SS3)"); self.push_command(AnsiCommand::EscapeSequence('O', vec![])); self.state = ParserState::Ground; } // SS3
            b'P' => { trace!("Escape -> DCS"); self.state = ParserState::DCS; } // DCS (ESC P)
            b'_' => { trace!("Escape: Ignoring APC (ESC _)"); self.state = ParserState::Ground; } // APC (ESC _)
            b'^' => { trace!("Escape: Ignoring PM (ESC ^)"); self.state = ParserState::Ground; } // PM (ESC ^)
            b'(' | b')' | b'*' | b'+' => { // SCS Start (Select Character Set)
                trace!("Escape -> Escape (charset designation started with '{}')", byte as char);
                // Remain in Escape state, waiting for the character set identifier
                // TODO: Handle charset sequences properly, maybe store the introducer byte
                self.csi_intermediates.push(byte as char); // Misusing csi_intermediates for now
                self.state = ParserState::Escape;
            }
            b'D' => { trace!("Escape -> Ground (IND)"); self.push_command(AnsiCommand::Csi(CsiCommand::CursorNextLine(1))); self.state = ParserState::Ground; } // IND (ESC D)
            b'E' => { trace!("Escape -> Ground (NEL)"); self.push_command(AnsiCommand::Csi(CsiCommand::CursorNextLine(1))); self.state = ParserState::Ground; } // NEL (ESC E)
            b'H' => { trace!("Escape -> Ground (HTS)"); self.push_command(AnsiCommand::Csi(CsiCommand::SetTabStop)); self.state = ParserState::Ground; } // HTS (ESC H)
            b'M' => { trace!("Escape -> Ground (RI)"); self.push_command(AnsiCommand::Csi(CsiCommand::CursorPrevLine(1))); self.state = ParserState::Ground; } // RI (ESC M)
            b'Z' => { debug!("Escape -> Ground (DECID) - Not Implemented"); self.state = ParserState::Ground; } // DECID (ESC Z)
            b'c' => { debug!("Escape -> Ground (RIS) - Not Implemented as command yet"); self.state = ParserState::Ground; } // RIS (ESC c)
            b'7' => { trace!("Escape -> Ground (DECSC)"); self.push_command(AnsiCommand::Csi(CsiCommand::SaveCursor)); self.state = ParserState::Ground; } // DECSC (ESC 7)
            b'8' => { trace!("Escape -> Ground (DECRC)"); self.push_command(AnsiCommand::Csi(CsiCommand::RestoreCursor)); self.state = ParserState::Ground; } // DECRC (ESC 8)
            b'=' => { debug!("Escape -> Ground (DECPAM) - Not Implemented"); self.state = ParserState::Ground; } // DECPAM (ESC =)
            b'>' => { debug!("Escape -> Ground (DECPNM) - Not Implemented"); self.state = ParserState::Ground; } // DECPNM (ESC >)

            // Handle charset selection characters after SCS start (e.g., ESC ( B)
            // This is a simplified handling - a full implementation needs to track which G-set is being designated.
            b'B' | b'0' | b'A' | b'K' => { // Basic charset identifiers
                 if !self.csi_intermediates.is_empty() { // Check if we were expecting a charset identifier
                    let introducer = self.csi_intermediates.pop().unwrap(); // Get the SCS introducer
                    trace!("Escape -> Ground (SCS {} {}) - Not Implemented as command yet", introducer as char, byte as char);
                    // TODO: Generate a proper SetCharset command
                 } else {
                     warn!("Escape: Received unexpected byte 0x{:02X} ('{}') after ESC", byte, if byte.is_ascii() { byte as char } else { '.' });
                 }
                 self.state = ParserState::Ground;
                 self.csi_intermediates.clear(); // Clear intermediates used for SCS
            }

            _ => { // Unknown ESC sequence
                debug!("Escape -> Ground: Ignoring unknown byte after ESC: 0x{:02X} ('{}')", byte, if byte.is_ascii() { byte as char } else { '.' });
                self.push_command(AnsiCommand::Ignore(byte));
                self.state = ParserState::Ground;
                 self.csi_intermediates.clear(); // Clear any intermediates
            }
        }
    }

    /// Handles the first byte after CSI (ESC [ or C1 0x9B) has been received.
    fn handle_csi_entry(&mut self, byte: u8) {
        match byte {
            b'0'..=b'9' => { // Parameter bytes
                trace!("CSIEntry -> CSIParam");
                self.state = ParserState::CSIParam;
                self.push_csi_param((byte - b'0') as u16);
            }
            b';' => { // Parameter separator
                trace!("CSIEntry -> CSIParam (leading ';')");
                self.state = ParserState::CSIParam;
                self.next_csi_param();
            }
            b'?' | b'>' | b'=' => { // Private marker
                trace!("CSIEntry: Processing private marker '{}'", byte as char);
                if self.csi_intermediates.is_empty() {
                     self.csi_intermediates.push(byte as char);
                } else {
                     // Allow multiple private markers? st.c seems to only store the first.
                     // Let's stick to storing only the first for now.
                     warn!("Multiple private markers encountered in CSI sequence, ignoring subsequent ones.");
                }
                // Stay in CSIEntry after a private marker, waiting for params or final byte
                self.state = ParserState::CSIEntry;
            }
            0x20..=0x2F => { // Intermediate bytes
                trace!("CSIEntry -> CSIIntermediate");
                if self.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                    self.csi_intermediates.push(byte as char);
                    self.state = ParserState::CSIIntermediate;
                } else {
                    debug!("Max CSI intermediates ({}) reached, ignoring '{}', entering Ignore state", MAX_CSI_INTERMEDIATES, byte as char);
                    self.state = ParserState::CSIIgnore;
                }
            }
            0x40..=0x7E => { // Final bytes
                trace!("CSIEntry -> Ground (direct final byte '{}')", byte as char);
                self.csi_dispatch(byte); // Dispatch the command
                self.state = ParserState::Ground;
                self.clear_csi_params();
            }
            ESC => { // Abort on ESC
                 warn!("CSI sequence aborted by ESC");
                 self.push_command(AnsiCommand::Error("CSI sequence aborted by ESC".into()));
                 self.clear_csi_params();
                 self.state = ParserState::Escape;
            }
            // Abort on C1 controls
            0x80..=0x9F => {
                 warn!("CSI sequence aborted by C1 control: 0x{:02X}", byte);
                 self.push_command(AnsiCommand::Error(format!("CSI sequence aborted by C1 control: 0x{:02X}", byte)));
                 self.clear_csi_params();
                 self.state = ParserState::Ground;
            }
            // Abort on unexpected C0 controls
            NUL..=BEL | SO..=SUB | 0x1C..=US | DEL => {
                 warn!("CSI sequence aborted by unexpected C0 control: 0x{:02X}", byte);
                 self.push_command(AnsiCommand::Error(format!("CSI sequence aborted by unexpected C0 control: 0x{:02X}", byte)));
                 self.clear_csi_params();
                 self.state = ParserState::Ground;
            }
            // Unexpected bytes - Abort sequence
            _ => {
                warn!("Unexpected byte 0x{:02X} in CSIEntry state, aborting sequence.", byte);
                 self.push_command(AnsiCommand::Error(format!("Unexpected byte 0x{:02X} in CSIEntry state", byte)));
                self.clear_csi_params();
                self.state = ParserState::Ground;
            }
        }
    }

    /// Handles CSI parameter bytes (digits, ';').
    fn handle_csi_param(&mut self, byte: u8) {
        match byte {
            b'0'..=b'9' => { // Parameter bytes
                self.push_csi_param((byte - b'0') as u16);
                self.state = ParserState::CSIParam; // Stay
            }
            b';' => { // Parameter separator
                self.next_csi_param();
                self.state = ParserState::CSIParam; // Stay
            }
            b'?' | b'>' | b'=' => { // Private marker (defensive)
                // According to ECMA-48, parameters should not contain these bytes.
                // If we receive them here, it's likely an error or non-standard sequence.
                warn!("Received private marker '{}' after CSI parameters, ignoring.", byte as char);
                // We could transition to CSIIgnore, but staying might allow recovery
                // if the final byte is valid. Let's stay for now.
                self.state = ParserState::CSIParam;
            }
            0x20..=0x2F => { // Intermediate bytes
                trace!("CSIParam -> CSIIntermediate");
                if self.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                    self.csi_intermediates.push(byte as char);
                    self.state = ParserState::CSIIntermediate;
                } else {
                    debug!("Max CSI intermediates ({}) reached, ignoring '{}', entering Ignore state", MAX_CSI_INTERMEDIATES, byte as char);
                    self.state = ParserState::CSIIgnore;
                }
            }
            0x40..=0x7E => { // Final bytes
                trace!("CSIParam -> Ground (final byte '{}')", byte as char);
                self.csi_dispatch(byte); // Dispatch the command
                self.state = ParserState::Ground;
                self.clear_csi_params();
            }
             ESC => { // Abort on ESC
                  warn!("CSI sequence aborted by ESC");
                  self.push_command(AnsiCommand::Error("CSI sequence aborted by ESC".into()));
                  self.clear_csi_params();
                  self.state = ParserState::Escape;
             }
             // Abort on C1 controls
             0x80..=0x9F => {
                  warn!("CSI sequence aborted by C1 control: 0x{:02X}", byte);
                  self.push_command(AnsiCommand::Error(format!("CSI sequence aborted by C1 control: 0x{:02X}", byte)));
                  self.clear_csi_params();
                  self.state = ParserState::Ground;
             }
             // Abort on unexpected C0 controls
             NUL..=BEL | SO..=SUB | 0x1C..=US | DEL => {
                  warn!("CSI sequence aborted by unexpected C0 control: 0x{:02X}", byte);
                  self.push_command(AnsiCommand::Error(format!("CSI sequence aborted by unexpected C0 control: 0x{:02X}", byte)));
                  self.clear_csi_params();
                  self.state = ParserState::Ground;
             }
             // Unexpected bytes - Abort sequence
             _ => {
                 warn!("Unexpected byte 0x{:02X} in CSIParam state, aborting sequence.", byte);
                 self.push_command(AnsiCommand::Error(format!("Unexpected byte 0x{:02X} in CSIParam state", byte)));
                 self.clear_csi_params();
                 self.state = ParserState::Ground;
             }
        }
    }

    /// Handles CSI intermediate bytes (after params, before final byte).
    fn handle_csi_intermediate(&mut self, byte: u8) {
         trace!("Entering handle_csi_intermediate_byte with byte 0x{:02X}", byte);
         match byte {
             0x20..=0x2F => { // Intermediate bytes
                 if self.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                     self.csi_intermediates.push(byte as char);
                     trace!("  -> Pushed intermediate '{}', current intermediates: {:?}", byte as char, self.csi_intermediates);
                     self.state = ParserState::CSIIntermediate; // Stay
                     trace!("  -> State explicitly set to: {:?}", self.state);
                 } else {
                     debug!("Max CSI intermediates ({}) reached, ignoring '{}', entering Ignore state", MAX_CSI_INTERMEDIATES, byte as char);
                     self.state = ParserState::CSIIgnore;
                 }
             }
             0x40..=0x7E => { // Final bytes
                 trace!("CSIIntermediate -> Ground (final byte '{}')", byte as char);
                 self.csi_dispatch(byte); // Dispatch the command
                 self.state = ParserState::Ground;
                 self.clear_csi_params();
             }
             ESC => { // Abort on ESC
                  warn!("CSI sequence aborted by ESC");
                  self.push_command(AnsiCommand::Error("CSI sequence aborted by ESC".into()));
                  self.clear_csi_params();
                  self.state = ParserState::Escape;
             }
             // Abort on C1 controls
             0x80..=0x9F => {
                  warn!("CSI sequence aborted by C1 control: 0x{:02X}", byte);
                  self.push_command(AnsiCommand::Error(format!("CSI sequence aborted by C1 control: 0x{:02X}", byte)));
                  self.clear_csi_params();
                  self.state = ParserState::Ground;
             }
             // Abort on unexpected C0 controls
             NUL..=BEL | SO..=SUB | 0x1C..=US | DEL => {
                  warn!("CSI sequence aborted by unexpected C0 control: 0x{:02X}", byte);
                  self.push_command(AnsiCommand::Error(format!("CSI sequence aborted by unexpected C0 control: 0x{:02X}", byte)));
                  self.clear_csi_params();
                  self.state = ParserState::Ground;
             }
             // Unexpected bytes (including 0x30-0x3F which are params/private markers)
             // ECMA-48 says these should be ignored in Intermediate state.
             // Let's transition to Ignore state to be safe and consume until final byte.
             _ => {
                 warn!("Unexpected byte 0x{:02X} in CSIIntermediate state, entering Ignore state.", byte);
                 self.push_command(AnsiCommand::Error(format!("Unexpected byte 0x{:02X} in CSIIntermediate state", byte)));
                 self.state = ParserState::CSIIgnore;
                 // Do not clear params here, CSIIgnore will clear on final byte
             }
         }
         trace!("Exiting handle_csi_intermediate_byte, state is {:?}", self.state);
    }

    /// Handles bytes when ignoring the rest of a CSI sequence.
    /// Consumes bytes until a final byte (0x40-0x7E) is found, then returns to Ground.
    fn handle_csi_ignore(&mut self, byte: u8) {
         match byte {
             0x40..=0x7E => { // Final bytes - Terminate the ignored sequence
                 trace!("CSIIgnore -> Ground (final byte '{}')", byte as char);
                 self.state = ParserState::Ground;
                 self.clear_csi_params();
             }
              ESC => { // Abort on ESC
                   warn!("Ignored CSI sequence aborted by ESC");
                   self.push_command(AnsiCommand::Error("Ignored CSI sequence aborted by ESC".into()));
                   self.clear_csi_params();
                   self.state = ParserState::Escape;
              }
              // Abort on C1 controls
              0x80..=0x9F => {
                   warn!("Ignored CSI sequence aborted by C1 control: 0x{:02X}", byte);
                   self.push_command(AnsiCommand::Error(format!("Ignored CSI sequence aborted by C1 control: 0x{:02X}", byte)));
                   self.clear_csi_params();
                   self.state = ParserState::Ground;
              }
             // Ignore other C0 controls within sequence
             NUL..=BEL | SO..=SUB | 0x1C..=US | DEL => {
                  trace!("CSIIgnore: Ignoring C0 control 0x{:02X}", byte);
             }
             // Ignore all other bytes while in this state
             _ => {
                  trace!("CSIIgnore: Ignoring byte 0x{:02X}", byte);
             }
         }
    }

    /// Handles bytes after OSC (ESC ]) or C1 OSC (0x9D) has been received.
    /// Collects bytes into `osc_string` until a terminator (BEL or ST) is found.
    fn handle_osc_string(&mut self, byte: u8) {
        match byte {
            BEL | ST_C1 => { // BEL or C1 ST (0x9C)
                trace!("OSCString -> Ground ({})", if byte == BEL { "BEL" } else { "ST_C1" });
                self.handle_osc_dispatch(); // Dispatch the OSC command
                self.state = ParserState::Ground;
                self.clear_osc_string();
            }
            ESC => { // ESC potentially starts ST (ESC \) or aborts
                 trace!("OSCString -> Escape (potential ST or abort)");
                 self.clear_osc_string(); // Abort current OSC string
                 self.state = ParserState::Escape;
            }
            // Ignore other C0/C1 controls within OSC string
            NUL..=ACK | BS..=SUB | 0x1C..=US | DEL | 0x80..=0x9B | 0x9D..=0x9F => {
                 trace!("OSCString: Ignoring control byte 0x{:02X}", byte);
            }
            // Collect all other bytes
            _ => {
                if self.osc_string.len() < MAX_OSC_STRING_LEN {
                    self.osc_string.push(byte);
                } else {
                     trace!("Maximum OSC string length ({}) exceeded, ignoring byte 0x{:02X}", MAX_OSC_STRING_LEN, byte);
                }
                self.state = ParserState::OSCString; // Stay
            }
        }
    }

    /// Handles bytes when parsing a DCS sequence. (Placeholder)
    fn handle_dcs(&mut self, byte: u8) {
        match byte {
            ST_C1 => { // C1 ST (0x9C) terminates DCS
                trace!("DCS -> Ground (ST_C1)");
                // TODO: Dispatch DCS command
                self.push_command(AnsiCommand::Dcs(vec![])); // Placeholder
                self.state = ParserState::Ground;
                // TODO: Clear DCS buffer
            }
            ESC => { // ESC potentially starts ST (ESC \) or aborts
                 trace!("DCS -> Escape (potential ST or abort)");
                 // TODO: Clear DCS buffer
                 self.state = ParserState::Escape;
            }
            // Collect other bytes
            _ => {
                // TODO: Collect DCS bytes
                trace!("DCS: Collecting byte 0x{:02X}", byte);
            }
        }
    }


    // --- Helper functions for CSI parameter handling (Adapted from original parser.rs) ---

    /// Clears CSI parameters and intermediates.
    fn clear_csi_params(&mut self) {
        self.csi_params.clear();
        self.csi_intermediates.clear();
    }

    /// Clears OSC string buffer.
    fn clear_osc_string(&mut self) {
        self.osc_string.clear();
    }

    /// Appends a digit to the last CSI parameter being built.
    fn push_csi_param(&mut self, digit: u16) {
        if self.csi_params.is_empty() {
            self.csi_params.push(digit);
            return;
        }
        if let Some(current_param) = self.csi_params.last_mut() {
            // Use saturating_mul and saturating_add to prevent overflow
            *current_param = current_param.saturating_mul(10).saturating_add(digit);
            if *current_param == u16::MAX {
                 warn!("CSI parameter overflow, clamped to {}", u16::MAX);
            }
        } else {
            // This case should ideally not happen if the vector is non-empty.
            warn!("push_csi_param: Attempted to push to non-existent parameter slot.");
        }
    }

    /// Moves to the next CSI parameter slot by appending a 0.
    fn next_csi_param(&mut self) {
        if self.csi_params.is_empty() {
            // Handle cases like CSI ; Pn... by adding an initial 0
            self.csi_params.push(0);
        }
        if self.csi_params.len() < MAX_CSI_PARAMS {
            self.csi_params.push(0);
        } else {
             trace!("Max CSI params ({}) reached, ignoring separator ';'", MAX_CSI_PARAMS);
        }
    }

     /// Gets the nth CSI parameter (0-based index), defaulting if absent.
     /// **Important:** This distinguishes between a missing parameter (returns `default`)
     /// and an explicitly provided parameter of 0 (returns `0`).
     fn get_csi_param(&self, index: usize, default: u16) -> u16 {
        self.csi_params.get(index).copied().unwrap_or(default)
    }

     /// Gets the nth CSI parameter (0-based index), returning 0 if absent.
     fn get_csi_param_or_0(&self, index: usize) -> u16 {
        self.csi_params.get(index).copied().unwrap_or(0)
    }


    // --- Dispatch Functions (Adapted from original parser.rs to return Commands) ---

    /// Dispatches a completed CSI sequence based on the final byte and parameters,
    /// generating a CsiCommand enum.
    fn csi_dispatch(&mut self, final_byte: u8) {
        let is_private = matches!(self.csi_intermediates.first(), Some(&'?') | Some(&'>') | Some(&'='));
        // Use p1_or_0 for operations where 0 is meaningful and distinct from missing/default 1
        let p1_or_0 = self.get_csi_param_or_0(0);
        // Use p1 for operations where 0/missing defaults to 1
        let p1 = self.get_csi_param(0, 1);
        // Use p2 for operations where 0/missing defaults to 1
        let p2 = self.get_csi_param(1, 1);

        trace!("Dispatch CSI: final='{}', params={:?}, intermediates={:?}, private={}, p1_or_0={}, p1={}, p2={}",
               final_byte as char, self.csi_params, self.csi_intermediates, is_private, p1_or_0, p1, p2);

        let command = if is_private {
            // Handle private sequences (DECSET/DECRST)
            match final_byte {
                b'h' => { // DECSET
                    // Iterate through all parameters for DECSET/DECRST
                    let modes: Vec<DecPrivateMode> = self.csi_params.iter()
                        .map(|&p| DecPrivateMode::from_param(p))
                        .collect();
                    // Note: A single CSI sequence can set/reset multiple modes
                    // We could return a single command with a list of modes,
                    // or push multiple commands. Pushing multiple commands
                    // simplifies the interpreter's job.
                    for mode in modes {
                        self.push_command(AnsiCommand::Csi(CsiCommand::DecPrivateModeSet(mode)));
                    }
                    return; // Return early as commands are pushed individually
                }
                b'l' => { // DECRST
                    let modes: Vec<DecPrivateMode> = self.csi_params.iter()
                        .map(|&p| DecPrivateMode::from_param(p))
                        .collect();
                    for mode in modes {
                        self.push_command(AnsiCommand::Csi(CsiCommand::DecPrivateModeReset(mode)));
                    }
                    return; // Return early
                }
                _ => {
                    warn!("Unhandled private CSI sequence: final='{}'", final_byte as char);
                    CsiCommand::UnhandledCsi {
                        intermediates: self.csi_intermediates.clone(),
                        final_byte: final_byte as char,
                        params: self.csi_params.clone(),
                    }
                }
            }
        } else {
            // Handle standard sequences
            match final_byte {
                // --- Cursor Movement ---
                b'A' => CsiCommand::CursorUp(p1),       // CUU
                b'B' => CsiCommand::CursorDown(p1),     // CUD
                b'C' => CsiCommand::CursorForward(p1),  // CUF
                b'D' => CsiCommand::CursorBackward(p1), // CUB
                b'E' => CsiCommand::CursorNextLine(p1), // CNL
                b'F' => CsiCommand::CursorPrevLine(p1), // CPL
                b'G' => CsiCommand::CursorHorizontalAbsolute(p1), // CHA
                b'H' | b'f' => CsiCommand::CursorPosition { row: p1, col: p2 }, // CUP / HVP
                b'd' => CsiCommand::VerticalLineAbsolute(p1), // VPA

                // --- Erasing ---
                // WHY: Use p1_or_0 because ED/EL param 0 is meaningful & distinct from default 1
                b'J' => CsiCommand::EraseDisplay(EraseMode::from_param(p1_or_0)), // ED
                b'K' => CsiCommand::EraseLine(EraseMode::from_param(p1_or_0)),    // EL
                b'X' => CsiCommand::EraseCharacter(p1), // ECH
                b'P' => CsiCommand::DeleteCharacter(p1), // DCH

                // --- Scrolling ---
                b'S' => CsiCommand::ScrollUp(p1),   // SU
                b'T' => CsiCommand::ScrollDown(p1), // SD

                // --- Insertion/Deletion ---
                b'L' => CsiCommand::InsertLine(p1), // IL
                b'M' => CsiCommand::DeleteLine(p1), // DL
                b'@' => CsiCommand::InsertCharacter(p1), // ICH

                // --- Tabs ---
                b'I' => CsiCommand::CursorForwardTab(p1), // CHT
                b'Z' => CsiCommand::CursorBackwardTab(p1), // CBT
                // WHY: Use p1_or_0 because TBC param 0 is meaningful
                b'g' => CsiCommand::TabClear(p1_or_0), // TBC

                // --- Attributes ---
                b'm' => {
                    // SGR can have multiple parameters, need to parse them all
                    let mut sgr_params = Vec::new();
                    let mut i = 0;
                    while i < self.csi_params.len() {
                        let param = self.csi_params[i];
                        sgr_params.push(SgrParameter::from_param(param, &self.csi_params, &mut i));
                        i += 1; // from_param increments i, but the loop also increments, so adjust
                    }
                     // The last increment in the loop needs to be undone if from_param
                     // already advanced i past the current parameter. The SgrParameter::from_param
                     // implementation handles its own index advancement, so the outer loop
                     // should just increment by 1 each time. Let's adjust SgrParameter::from_param
                     // to not take and modify `current_idx` and handle SGR parsing here.
                     // Let's redo SGR parsing here to be clearer.

                     let mut sgr_commands = Vec::new();
                     let mut i = 0;
                     while i < self.csi_params.len() {
                         let param = self.csi_params[i];
                         match param {
                             0 => sgr_commands.push(SgrParameter::Reset),
                             1 => sgr_commands.push(SgrParameter::Bold),
                             2 => sgr_commands.push(SgrParameter::Faint),
                             3 => sgr_commands.push(SgrParameter::Italic),
                             4 => sgr_commands.push(SgrParameter::Underline),
                             5 | 6 => sgr_commands.push(SgrParameter::Blink),
                             7 => sgr_commands.push(SgrParameter::Reverse),
                             8 => sgr_commands.push(SgrParameter::Hidden),
                             9 => sgr_commands.push(SgrParameter::Strikethrough),
                             22 => sgr_commands.push(SgrParameter::NormalIntensity),
                             23 => sgr_commands.push(SgrParameter::ItalicOff),
                             24 => sgr_commands.push(SgrParameter::UnderlineOff),
                             25 => sgr_commands.push(SgrParameter::BlinkOff),
                             27 => sgr_commands.push(SgrParameter::ReverseOff),
                             28 => sgr_commands.push(SgrParameter::HiddenOff),
                             29 => sgr_commands.push(SgrParameter::StrikethroughOff),
                             30..=37 => sgr_commands.push(SgrParameter::ForegroundBasic((param - 30) as u8)),
                             40..=47 => sgr_commands.push(SgrParameter::BackgroundBasic((param - 40) as u8)),
                             39 => sgr_commands.push(SgrParameter::ForegroundDefault),
                             49 => sgr_commands.push(SgrParameter::BackgroundDefault),
                             38 | 48 => { // Extended colors
                                 let is_fg = param == 38;
                                 i += 1; // Consume the specifier parameter (5 or 2)
                                 if i >= self.csi_params.len() {
                                     warn!("SGR: Missing specifier after {}", param);
                                     sgr_commands.push(SgrParameter::Unknown(param));
                                     break;
                                 }
                                 let specifier = self.csi_params[i];

                                 match specifier {
                                     5 => { // 256-color mode
                                         i += 1; // Consume the color index parameter
                                         if i >= self.csi_params.len() {
                                             warn!("SGR: Missing color index after {} ; 5", param);
                                             sgr_commands.push(SgrParameter::Unknown(param));
                                             break;
                                         }
                                         let color_idx = self.csi_params[i];
                                         if color_idx <= 255 {
                                             if is_fg { sgr_commands.push(SgrParameter::Foreground256Color(color_idx as u8)) }
                                             else { sgr_commands.push(SgrParameter::Background256Color(color_idx as u8)) }
                                         } else {
                                             warn!("SGR: Invalid 256-color index: {}", color_idx);
                                             sgr_commands.push(SgrParameter::Unknown(color_idx));
                                         }
                                     }
                                     2 => { // Truecolor (RGB) mode
                                         if i + 3 >= self.csi_params.len() {
                                             warn!("SGR: Missing RGB values after {} ; 2", param);
                                             sgr_commands.push(SgrParameter::Unknown(param));
                                             break;
                                         }
                                         let r = self.csi_params[i+1].min(255) as u8;
                                         let g = self.csi_params[i+2].min(255) as u8;
                                         let b = self.csi_params[i+3].min(255) as u8;
                                         i += 3; // Consume R, G, B
                                         if is_fg { sgr_commands.push(SgrParameter::ForegroundRgb(r, g, b)) }
                                         else { sgr_commands.push(SgrParameter::BackgroundRgb(r, g, b)) }
                                     }
                                     _ => {
                                         warn!("SGR: Invalid specifier {} after {}", specifier, param);
                                         sgr_commands.push(SgrParameter::Unknown(specifier));
                                     }
                                 }
                             }
                             _ => {
                                 warn!("Unhandled SGR parameter: {}", param);
                                 sgr_commands.push(SgrParameter::Unknown(param));
                             }
                         }
                         i += 1;
                     }
                     CsiCommand::SetGraphicsRendition(sgr_commands)
                }

                // --- Modes ---
                // Standard modes are often ignored by terminal emulators as they relate
                // to the physical terminal hardware. We'll generate commands for them
                // but the interpreter might choose to ignore them.
                b'h' => { warn!("Ignoring standard Set Mode (SM) sequence: CSI {} h", p1_or_0); CsiCommand::UnhandledCsi { intermediates: self.csi_intermediates.clone(), final_byte: final_byte as char, params: self.csi_params.clone() } }
                b'l' => { warn!("Ignoring standard Reset Mode (RM) sequence: CSI {} l", p1_or_0); CsiCommand::UnhandledCsi { intermediates: self.csi_intermediates.clone(), final_byte: final_byte as char, params: self.csi_params.clone() } }

                // --- Reporting ---
                // DSR and DA commands might require the terminal to send data back
                // to the PTY. The interpreter will need access to the PTY writer.
                // We'll generate the command, and the interpreter handles the response.
                b'n' => CsiCommand::DeviceStatusReport(p1_or_0), // DSR
                b'c' => CsiCommand::DeviceAttributes(p1_or_0), // DA

                // --- Scrolling Region ---
                b'r' => CsiCommand::SetScrollingRegion { top: p1, bottom: p2 }, // DECSTBM

                // --- Cursor Save/Restore ---
                b's' => CsiCommand::SaveCursor,   // SCOSC
                b'u' => CsiCommand::RestoreCursor, // SCORC

                _ => { // Unhandled standard sequence
                    warn!(
                        "Unhandled standard CSI sequence: final='{}' (0x{:02X}), params={:?}, intermediates={:?}",
                        final_byte as char, final_byte, self.csi_params, self.csi_intermediates
                    );
                    CsiCommand::UnhandledCsi {
                        intermediates: self.csi_intermediates.clone(),
                        final_byte: final_byte as char,
                        params: self.csi_params.clone(),
                    }
                }
            }
        };

        // Push the generated CSI command to the queue
        self.push_command(AnsiCommand::Csi(command));
    }

    /// Dispatches a completed OSC sequence based on the collected string,
    /// generating an OscCommand enum.
    fn handle_osc_dispatch(&mut self) {
        let osc_content = String::from_utf8_lossy(&self.osc_string);
        trace!("Dispatching OSC: '{}'", osc_content);

        let mut parts = osc_content.splitn(2, ';');
        let code_str = parts.next().unwrap_or("");
        let arg = parts.next().unwrap_or("");

        if code_str.is_empty() {
            warn!("Empty OSC code received.");
            self.push_command(AnsiCommand::Error("Empty OSC code received".into()));
            return;
        }

        let command = match code_str {
            "0" => OscCommand::SetIconAndWindowTitle(arg.to_string()),
            "1" => OscCommand::SetIconName(arg.to_string()),
            "2" => OscCommand::SetWindowTitle(arg.to_string()),
            _ => {
                warn!("Unhandled OSC sequence code: '{}', arg: '{}'", code_str, arg);
                // Attempt to parse the code as an integer for the Unhandled variant
                let command_num = code_str.parse::<i32>().unwrap_or(-1);
                OscCommand::UnhandledOsc { command: command_num, data: arg.to_string() }
            }
        };

        // Push the generated OSC command to the queue
        self.push_command(AnsiCommand::Osc(command));
    }
}

// --- Utf8Receiver Implementation ---

/// Helper struct to receive characters from the Utf8Parser and push them as Commands.
struct Utf8CommandReceiver<'a> {
    parser: &'a mut AnsiParser,
}

impl<'a> Utf8Receiver for Utf8CommandReceiver<'a> {
    fn codepoint(&mut self, c: char) {
        // Only push printable characters when in the Ground state.
        // Control characters and sequence introducers are handled by the main state machine.
        if self.parser.state == ParserState::Ground {
             self.parser.push_command(AnsiCommand::Print(c));
        } else {
            // If we receive a codepoint while in a sequence state, it's likely an error
            // or unexpected data within a sequence (like text in a DCS).
            // For now, we'll ignore it or handle it based on the specific state's needs
            // (e.g., OSCString collects bytes, not chars from Utf8Parser).
             trace!("UTF8 receiver: Received codepoint '{}' in state {:?}", c, self.parser.state);
             // Depending on the state, you might push an Ignore or Error command here.
             // For simplicity, we'll let the main state machine handle byte-by-byte
             // processing for sequence states.
        }
    }

    fn invalid_sequence(&mut self) {
        // Handle invalid UTF-8 sequences. Replace with a replacement character or similar.
        warn!("Received invalid UTF-8 sequence");
        if self.parser.state == ParserState::Ground {
            // Push the replacement character if we are in the Ground state
            self.parser.push_command(AnsiCommand::Print(std::char::REPLACEMENT_CHARACTER));
        } else {
            // If invalid UTF-8 occurs within a sequence, it might indicate a problem.
            // Depending on the state, you might want to push an Error command.
            trace!("UTF8 receiver: Invalid sequence in state {:?}", self.parser.state);
        }
    }
}


// --- Implementations for C0Control (Copied and adapted from original parser.rs) ---

impl C0Control {
    fn from_byte(byte: u8) -> Self {
        match byte {
            NUL => C0Control::Null,
            BEL => C0Control::Bell,
            BS => C0Control::Backspace,
            HT => C0Control::Tab,
            LF => C0Control::LineFeed,
            VT => C0Control::VerticalTab,
            FF => C0Control::FormFeed,
            CR => C0Control::CarriageReturn,
            SO => C0Control::ShiftOut,
            SI => C0Control::ShiftIn,
            CAN => C0Control::Cancel,
            SUB => C0Control::Substitute,
            ESC => C0Control::Escape, // Should ideally not be handled here in Ground state
            DEL => C0Control::Delete,
            // Add other C0 controls if needed
            _ => C0Control::Unknown(byte),
        }
    }
}

#[cfg(test)]
mod tests;
