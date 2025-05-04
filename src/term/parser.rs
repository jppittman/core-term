// src/term/parser.rs

//! Handles parsing of input byte streams, including escape sequences (CSI, OSC).

// Note: `Term` methods related to screen manipulation (scrolling, erasing, cursor movement, etc.)
// will be called from here but defined in `screen.rs`.
use super::{Term, ParserState}; // Import Term and ParserState from parent module (mod.rs)
use crate::glyph::{Color, AttrFlags};
use super::{MAX_CSI_PARAMS, MAX_CSI_INTERMEDIATES, MAX_OSC_STRING_LEN}; // Import constants
use std::cmp::min;
// Import logging macros
use log::{trace, debug, warn};

// --- Constants for Control Codes ---
// C0 Controls
const NUL: u8 = 0x00;
const SOH: u8 = 0x01;
const STX: u8 = 0x02;
const ETX: u8 = 0x03;
const EOT: u8 = 0x04;
const ENQ: u8 = 0x05;
const ACK: u8 = 0x06;
const BEL: u8 = 0x07;
const BS: u8 = 0x08;
const HT: u8 = 0x09;
const LF: u8 = 0x0A;
const VT: u8 = 0x0B;
const FF: u8 = 0x0C;
const CR: u8 = 0x0D;
const SO: u8 = 0x0E;
const SI: u8 = 0x0F;
const DLE: u8 = 0x10;
const DC1: u8 = 0x11; // XON
const DC2: u8 = 0x12;
const DC3: u8 = 0x13; // XOFF
const DC4: u8 = 0x14;
const NAK: u8 = 0x15;
const SYN: u8 = 0x16;
const ETB: u8 = 0x17;
const CAN: u8 = 0x18;
const EM: u8 = 0x19;
const SUB: u8 = 0x1A;
const ESC: u8 = 0x1B;
const FS: u8 = 0x1C;
const GS: u8 = 0x1D;
const RS: u8 = 0x1E;
const US: u8 = 0x1F;
const DEL: u8 = 0x7F;

// C1 Controls (often represented as ESC + Fe)
const IND: u8 = 0x84; // ESC D
const NEL: u8 = 0x85; // ESC E
const HTS: u8 = 0x88; // ESC H
const RI: u8 = 0x8D;  // ESC M
const SS2: u8 = 0x8E; // ESC N
const SS3: u8 = 0x8F; // ESC O
const DCS: u8 = 0x90; // ESC P
const CSI: u8 = 0x9B; // ESC [
const ST: u8 = 0x9C;  // ESC \
const OSC: u8 = 0x9D; // ESC ]
const PM: u8 = 0x9E;  // ESC ^
const APC: u8 = 0x9F; // ESC _

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
const SGR_FG_OFFSET: u16 = 30;
const SGR_BG_OFFSET: u16 = 40;
const SGR_FG_BRIGHT_OFFSET: u16 = 90;
const SGR_BG_BRIGHT_OFFSET: u16 = 100;
const SGR_FG_DEFAULT: u16 = 39;
const SGR_BG_DEFAULT: u16 = 49;
const SGR_EXTENDED_FG: u16 = 38;
const SGR_EXTENDED_BG: u16 = 48;
const SGR_EXTENDED_MODE_256: u16 = 5;
const SGR_EXTENDED_MODE_RGB: u16 = 2;

// --- Constants for DEC Private Modes (CSI ? Pn h/l) ---
const DECCKM: u16 = 1;   // Cursor Keys Mode
const DECANM: u16 = 2;   // ANSI/VT52 Mode
const DECCOLM: u16 = 3;  // 132 Column Mode
const DECSCLM: u16 = 4;  // Smooth Scroll
const DECSCNM: u16 = 5;  // Screen Mode (Reverse Video)
const DECOM: u16 = 6;    // Origin Mode
const DECAWM: u16 = 7;   // Autowrap Mode
const DECARM: u16 = 8;   // Autorepeat
const X10_MOUSE: u16 = 9; // X10 Mouse Reporting
const BLINK_CURSOR_ATT610: u16 = 12; // Blinking Cursor Enable Mode (att610)
const DECPFF: u16 = 18;  // Print Form Feed
const DECPEX: u16 = 19;  // Print Extent
const DECTCEM: u16 = 25; // Text Cursor Enable Mode (Show/Hide)
const DECNRCM: u16 = 42; // National Replacement Character sets
const ALT_SCREEN_BUF_47: u16 = 47; // Use Alternate Screen Buffer
const MOUSE_HIGHLIGHT: u16 = 1001; // Mouse Highlight Reporting
const MOUSE_UTF8: u16 = 1005; // UTF8 Mouse Reporting
const MOUSE_URXVT: u16 = 1015; // urxvt Mouse Mode
const MOUSE_BTN_REPORT: u16 = 1000; // Send Mouse X & Y on button press and release.
const MOUSE_BTN_MOTION_REPORT: u16 = 1002; // Send Mouse X & Y on button press and motion.
const MOUSE_ANY_MOTION_REPORT: u16 = 1003; // Send Mouse X & Y on any motion.
const FOCUS_EVENT_REPORT: u16 = 1004; // Send FocusIn/FocusOut events
const MOUSE_SGR_REPORT: u16 = 1006; // SGR Mouse Mode
const INPUT_MODE_8BIT: u16 = 1034; // Interpret "meta" key, sets eighth bit.
const ALT_SCREEN_BUF_1047: u16 = 1047; // Use Alternate Screen Buffer (like 47)
const CURSOR_SAVE_RESTORE_1048: u16 = 1048; // Save/Restore Cursor (like DECSC/DECRC)
const ALT_SCREEN_SAVE_RESTORE_1049: u16 = 1049; // Combines 1047 and 1048
const BRACKETED_PASTE: u16 = 2004; // Bracketed Paste Mode

// Helper array for ALT_SCREEN modes
const ALT_SCREEN_MODES: [u16; 3] = [ALT_SCREEN_BUF_47, ALT_SCREEN_BUF_1047, ALT_SCREEN_SAVE_RESTORE_1049];


// --- Parser State Handling Functions (taking `&mut Term`) ---

/// Clears CSI parameters and intermediates.
/// Should be called when entering ESC state or finishing/aborting a CSI sequence.
pub(super) fn clear_csi_params(term: &mut Term) {
    // trace!("Clearing CSI params and intermediates"); // Keep commented unless debugging parser
    term.csi_params.clear();
    term.csi_intermediates.clear();
}

/// Clears OSC string buffer.
/// Should be called when entering ESC state or finishing/aborting an OSC sequence.
pub(super) fn clear_osc_string(term: &mut Term) {
    // trace!("Clearing OSC string"); // Keep commented unless debugging parser
    term.osc_string.clear();
}

/// Appends a digit to the last CSI parameter being built.
/// Handles potential overflow and ignores digits if max parameters reached.
pub(super) fn push_csi_param(term: &mut Term, digit: u16) {
    // Ensure there's at least one parameter slot to work with.
    if term.csi_params.is_empty() {
        term.csi_params.push(digit);
        return;
    }

    // Get the last parameter.
    if let Some(current_param) = term.csi_params.last_mut() {
        *current_param = current_param.saturating_mul(10).saturating_add(digit);
        if *current_param == u16::MAX {
             warn!("CSI parameter overflow, clamped to {}", u16::MAX);
        }
    } else {
        warn!("push_csi_param: Attempted to push to non-existent parameter slot.");
    }
}

/// Moves to the next CSI parameter slot by appending a 0.
/// Does nothing if the maximum number of parameters has already been reached.
pub(super) fn next_csi_param(term: &mut Term) {
    if term.csi_params.is_empty() {
        term.csi_params.push(0);
    }

    if term.csi_params.len() < MAX_CSI_PARAMS {
        term.csi_params.push(0);
    } else {
         trace!("Max CSI params ({}) reached, ignoring separator ';'", MAX_CSI_PARAMS);
    }
}

 /// Gets the nth CSI parameter (0-based index), defaulting if absent or 0.
 /// Used when a parameter value of 0 should be treated as the default (often 1).
 pub(super) fn get_csi_param(term: &Term, index: usize, default: u16) -> u16 {
    match term.csi_params.get(index) {
        Some(&p) if p > 0 => p,
        _ => default,
    }
}

 /// Gets the nth CSI parameter (0-based index), returning 0 if absent.
 /// Used when a missing parameter should be treated as 0.
 pub(super) fn get_csi_param_or_0(term: &Term, index: usize) -> u16 {
    term.csi_params.get(index).copied().unwrap_or(0)
}


// --- State Transition Handlers ---

/// Handles C0 control codes and ESC when in the Ground state.
pub(super) fn handle_ground_control_or_esc(term: &mut Term, byte: u8) {
    match byte {
        // C0 Control Codes - Delegate to screen module for actions
        BS => super::screen::backspace(term),
        HT => super::screen::tab(term),
        LF | VT | FF => super::screen::newline(term),
        CR => super::screen::carriage_return(term),
        BEL => { debug!("BEL received (bell not implemented)"); }
        SO | SI => { debug!("SO/SI received (charset switching ignored)"); }
        ESC => {
            trace!("Ground -> Escape");
            term.parser_state = ParserState::Escape;
            clear_csi_params(term);
            clear_osc_string(term);
        }
        // C1 Control Codes (0x80 - 0x9F)
        IND => { super::screen::index(term); }
        NEL => { super::screen::newline(term); }
        HTS => { debug!("HTS received (C1 0x88 - Not Implemented)"); }
        RI => { super::screen::reverse_index(term); }
        CSI => {
            trace!("Ground -> CSIEntry (via C1 CSI)");
            clear_csi_params(term);
            term.parser_state = ParserState::CSIEntry;
        }
        OSC => {
             trace!("Ground -> OSCString (via C1 OSC)");
             clear_osc_string(term);
             term.parser_state = ParserState::OSCString;
        }
        ST => { trace!("Ignoring ST (0x9C) in Ground state"); }
        // Ignore other C0/C1 codes not explicitly handled
        NUL..=ACK | DLE..=SUB | FS..=US | DEL | // Other C0
        0x80..=0x83 | 0x86..=0x87 | 0x89..=0x8C | 0x8E..=0x9A | 0x9E..=0x9F // Other C1
        => {
             trace!("Ignoring C0/C1 control code: 0x{:02X}", byte);
        }
        _ => { warn!("handle_ground_control_or_esc called with unexpected byte: 0x{:02X}", byte); }
    }
}

/// Handles bytes after an ESC (0x1B) has been received.
pub(super) fn handle_escape_byte(term: &mut Term, byte: u8) {
    match byte {
        b'[' => {
            trace!("Escape -> CSIEntry");
            term.parser_state = ParserState::CSIEntry;
        }
        b']' => {
            trace!("Escape -> OSCString");
            term.parser_state = ParserState::OSCString;
        }
        b'\\' => { // ST (ESC \)
             trace!("Escape -> Ground (received ST)");
             term.parser_state = ParserState::Ground;
        }
        b'N' => { trace!("Ignoring SS2 (ESC N)"); term.parser_state = ParserState::Ground; } // SS2
        b'O' => { trace!("Ignoring SS3 (ESC O)"); term.parser_state = ParserState::Ground; } // SS3
        b'P' | b'_' | b'^' => { // DCS, APC, PM
             trace!("Escape -> OSCString (via DCS/APC/PM)");
             term.parser_state = ParserState::OSCString;
        }
        b'(' | b')' | b'*' | b'+' => { // SCS Start
            trace!("Escape -> Escape (charset designation started with '{}')", byte as char);
            term.parser_state = ParserState::Escape; // Remain in Escape
        }
        // --- Standard ESC sequences ---
        b'D' => { super::screen::index(term); term.parser_state = ParserState::Ground; } // IND
        b'E' => { super::screen::newline(term); term.parser_state = ParserState::Ground; } // NEL
        b'H' => { debug!("HTS received (ESC H - Not Implemented)"); term.parser_state = ParserState::Ground; } // HTS
        b'M' => { super::screen::reverse_index(term); term.parser_state = ParserState::Ground; } // RI
        b'Z' => { debug!("DECID received (ESC Z - Not Implemented)"); term.parser_state = ParserState::Ground; } // DECID
        b'c' => { super::screen::reset(term); term.parser_state = ParserState::Ground; } // RIS
        b'7' => { super::screen::save_cursor(term); term.parser_state = ParserState::Ground; } // DECSC
        b'8' => { super::screen::restore_cursor(term); term.parser_state = ParserState::Ground; } // DECRC
        b'=' => { debug!("DECPAM received (ESC = - Not Implemented)"); term.parser_state = ParserState::Ground; } // DECPAM
        b'>' => { debug!("DECPNM received (ESC > - Not Implemented)"); term.parser_state = ParserState::Ground; } // DECPNM

        // --- Charset selection (second byte after ESC ( ) * + ) ---
        b'B' => { trace!("Selected G0/G1/G2/G3 charset 'B' (US-ASCII)"); term.parser_state = ParserState::Ground; }
        b'0' => { trace!("Selected G0/G1/G2/G3 charset '0' (DEC Special Graphics)"); term.parser_state = ParserState::Ground; }

        _ => { // Unknown ESC sequence
            debug!("Ignoring unknown byte after ESC: 0x{:02X} ('{}')", byte, if byte.is_ascii() { byte as char } else { '.' });
            term.parser_state = ParserState::Ground;
        }
    }
}


/// Handles the first byte after CSI (ESC [) has been received.
pub(super) fn handle_csi_entry_byte(term: &mut Term, byte: u8) {
    match byte {
        b'0'..=b'9' => { // Parameter bytes
            trace!("CSIEntry -> CSIParam");
            term.parser_state = ParserState::CSIParam;
            push_csi_param(term, (byte - b'0') as u16);
        }
        b';' => { // Parameter separator
            trace!("CSIEntry -> CSIParam (leading ';')");
            term.parser_state = ParserState::CSIParam;
            next_csi_param(term);
        }
        b'?' | b'>' | b'=' => { // Private marker
            trace!("CSIEntry: Processing private marker '{}'", byte as char);
            if term.csi_intermediates.is_empty() {
                 term.csi_intermediates.push(byte as char);
            } else {
                 warn!("Multiple private markers encountered in CSI sequence");
            }
            term.parser_state = ParserState::CSIEntry; // Stay
        }
        0x20..=0x2F => { // Intermediate bytes
            trace!("CSIEntry -> CSIIntermediate");
            if term.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                term.csi_intermediates.push(byte as char);
                term.parser_state = ParserState::CSIIntermediate;
            } else {
                debug!("Max CSI intermediates ({}) reached, ignoring '{}', entering Ignore state", MAX_CSI_INTERMEDIATES, byte as char);
                term.parser_state = ParserState::CSIIgnore;
            }
        }
        0x40..=0x7E => { // Final bytes
            trace!("CSIEntry -> Ground (direct final byte '{}')", byte as char);
            csi_dispatch(term, byte);
            term.parser_state = ParserState::Ground;
            clear_csi_params(term);
        }
        ESC | 0x80..=0x9F => { // Abort on ESC or C1 controls
             warn!("CSI sequence aborted by ESC or C1 control: 0x{:02X}", byte);
             clear_csi_params(term);
             term.parser_state = ParserState::Ground;
        }
        NUL..=BEL | SO..=SUB | 0x1C..=US | DEL => { // Ignore other C0 controls
             trace!("CSIEntry: Ignoring C0 control 0x{:02X}", byte);
        }
        _ => { // Unexpected bytes - Abort sequence
            warn!("Unexpected byte 0x{:02X} in CSIEntry state, aborting sequence.", byte);
            clear_csi_params(term);
            term.parser_state = ParserState::Ground;
        }
    }
}

/// Handles CSI parameter bytes (digits, ';').
pub(super) fn handle_csi_param_byte(term: &mut Term, byte: u8) {
    match byte {
        b'0'..=b'9' => { // Parameter bytes
            push_csi_param(term, (byte - b'0') as u16);
            term.parser_state = ParserState::CSIParam; // Stay
        }
        b';' => { // Parameter separator
            next_csi_param(term);
            term.parser_state = ParserState::CSIParam; // Stay
        }
        b'?' | b'>' | b'=' => { // Private marker (defensive)
            warn!("Received private marker '{}' after CSI parameters, ignoring.", byte as char);
            term.parser_state = ParserState::CSIParam; // Stay
        }
        0x20..=0x2F => { // Intermediate bytes
            trace!("CSIParam -> CSIIntermediate");
            if term.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                term.csi_intermediates.push(byte as char);
                term.parser_state = ParserState::CSIIntermediate;
            } else {
                debug!("Max CSI intermediates ({}) reached, ignoring '{}', entering Ignore state", MAX_CSI_INTERMEDIATES, byte as char);
                term.parser_state = ParserState::CSIIgnore;
            }
        }
        0x40..=0x7E => { // Final bytes
            trace!("CSIParam -> Ground (final byte '{}')", byte as char);
            csi_dispatch(term, byte);
            term.parser_state = ParserState::Ground;
            clear_csi_params(term);
        }
         ESC | 0x80..=0x9F => { // Abort on ESC or C1 controls
              warn!("CSI sequence aborted by ESC or C1 control: 0x{:02X}", byte);
              clear_csi_params(term);
              term.parser_state = ParserState::Ground;
         }
         NUL..=BEL | SO..=SUB | 0x1C..=US | DEL => { // Ignore other C0 controls
              trace!("CSIParam: Ignoring C0 control 0x{:02X}", byte);
         }
         _ => { // Unexpected bytes - Abort sequence
             warn!("Unexpected byte 0x{:02X} in CSIParam state, aborting sequence.", byte);
             clear_csi_params(term);
             term.parser_state = ParserState::Ground;
         }
    }
}

/// Handles CSI intermediate bytes (after params, before final byte).
pub(super) fn handle_csi_intermediate_byte(term: &mut Term, byte: u8) {
     match byte {
         0x20..=0x2F => { // Intermediate bytes
             if term.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                 term.csi_intermediates.push(byte as char);
                 term.parser_state = ParserState::CSIIntermediate; // Stay
             } else {
                 debug!("Max CSI intermediates ({}) reached, ignoring '{}', entering Ignore state", MAX_CSI_INTERMEDIATES, byte as char);
                 term.parser_state = ParserState::CSIIgnore;
             }
         }
         0x40..=0x7E => { // Final bytes
             trace!("CSIIntermediate -> Ground (final byte '{}')", byte as char);
             csi_dispatch(term, byte);
             term.parser_state = ParserState::Ground;
             clear_csi_params(term);
         }
         ESC | 0x80..=0x9F => { // Abort on ESC or C1 controls
              warn!("CSI sequence aborted by ESC or C1 control: 0x{:02X}", byte);
              clear_csi_params(term);
              term.parser_state = ParserState::Ground;
         }
         NUL..=BEL | SO..=SUB | 0x1C..=US | DEL => { // Ignore other C0 controls
              trace!("CSIIntermediate: Ignoring C0 control 0x{:02X}", byte);
         }
         _ => { // Unexpected bytes - Abort sequence
             warn!("Unexpected byte 0x{:02X} in CSIIntermediate state, aborting sequence.", byte);
             clear_csi_params(term);
             term.parser_state = ParserState::Ground;
         }
     }
}

/// Handles bytes when ignoring the rest of a CSI sequence.
/// Consumes bytes until a final byte (0x40-0x7E) is found, then returns to Ground.
pub(super) fn handle_csi_ignore_byte(term: &mut Term, byte: u8) {
     match byte {
         0x40..=0x7E => { // Final bytes - Terminate the ignored sequence
             trace!("CSIIgnore -> Ground (final byte '{}')", byte as char);
             term.parser_state = ParserState::Ground;
             clear_csi_params(term);
         }
          ESC | 0x80..=0x9F => { // Abort on ESC or C1 controls
               warn!("Ignored CSI sequence aborted by ESC or C1 control: 0x{:02X}", byte);
               clear_csi_params(term);
               term.parser_state = ParserState::Ground;
          }
         NUL..=BEL | SO..=SUB | 0x1C..=US | DEL => { // Ignore other C0 controls
              trace!("CSIIgnore: Ignoring C0 control 0x{:02X}", byte);
         }
         _ => { // Ignore all other bytes while in this state
              trace!("CSIIgnore: Ignoring byte 0x{:02X}", byte);
         }
     }
}

/// Handles bytes after OSC (ESC ]) or C1 OSC (0x9D) has been received.
/// Collects bytes into `osc_string` until a terminator (BEL, ST) is found.
pub(super) fn handle_osc_string_byte(term: &mut Term, byte: u8) {
    match byte {
        BEL => { // BEL terminates OSC
            trace!("OSCString -> Ground (BEL terminator)");
            handle_osc_dispatch(term);
            term.parser_state = ParserState::Ground;
            clear_osc_string(term);
        }
        ESC => { // ESC potentially starts ST (ESC \)
             trace!("OSCString -> Escape (potential ST)");
             term.parser_state = ParserState::Escape;
             clear_osc_string(term); // Abort current OSC string
        }
         ST => { // ST (String Terminator) C1 form
              trace!("OSCString -> Ground (ST C1 terminator)");
              handle_osc_dispatch(term);
              term.parser_state = ParserState::Ground;
              clear_osc_string(term);
         }
        // Ignore other C0/C1 controls within OSC string
        NUL..=ACK | BS..=SUB | 0x1C..=US | DEL | 0x80..=0x9B | 0x9D..=0x9F => {
             trace!("OSCString: Ignoring control byte 0x{:02X}", byte);
        }
        // Collect all other bytes
        _ => {
            if term.osc_string.len() < MAX_OSC_STRING_LEN {
                // FIXME: Assumes byte is valid for String. Use Vec<u8> for safety.
                term.osc_string.push(byte as char);
            } else {
                 trace!("Maximum OSC string length ({}) exceeded, ignoring byte 0x{:02X}", MAX_OSC_STRING_LEN, byte);
            }
            term.parser_state = ParserState::OSCString; // Stay
        }
    }
}


// --- Dispatch Functions ---

/// Dispatches a completed CSI sequence based on the final byte.
fn csi_dispatch(term: &mut Term, final_byte: u8) {
    let is_private = matches!(term.csi_intermediates.first(), Some(&'?') | Some(&'>') | Some(&'='));
    let p1 = get_csi_param(term, 0, 1);
    let p2 = get_csi_param(term, 1, 1);
    let p1_or_0 = get_csi_param_or_0(term, 0);

    trace!("Dispatch CSI: final='{}', params={:?}, intermediates={:?}, private={}",
           final_byte as char, term.csi_params, term.csi_intermediates, is_private);

    // --- Dispatch Logic (using early return for private sequences) ---
    if is_private {
        match final_byte {
            b'h' => handle_dec_mode_enable(term),  // DECSET
            b'l' => handle_dec_mode_disable(term), // DECRST
            _ => warn!("Unhandled private CSI sequence: final='{}'", final_byte as char),
        }
        return; // Return early after handling private sequence
    }

    // --- Handle standard sequences (only reached if not private) ---
    match final_byte {
        // --- Cursor Movement ---
        b'A' => super::screen::move_cursor(term, 0, -(p1 as isize)), // CUU
        b'B' => super::screen::move_cursor(term, 0, p1 as isize),   // CUD
        b'C' => super::screen::move_cursor(term, get_csi_param(term, 0, 1) as isize, 0), // CUF
        b'D' => super::screen::move_cursor(term, -(get_csi_param(term, 0, 1) as isize), 0), // CUB
        b'E' => { // CNL
            super::screen::move_cursor(term, -(term.cursor.x as isize), p1 as isize);
        }
        b'F' => { // CPL
            super::screen::move_cursor(term, -(term.cursor.x as isize), -(p1 as isize));
        }
        b'G' => { // CHA
             let target_x = p1.saturating_sub(1) as usize;
             term.cursor.x = min(target_x, term.width.saturating_sub(1));
        }
        b'H' | b'f' => { // CUP / HVP
            let target_y = p1.saturating_sub(1) as usize;
            let target_x = p2.saturating_sub(1) as usize;
            super::screen::set_cursor_pos(term, target_x + 1, target_y + 1); // Pass 1-based
        }
        b'd' => { // VPA
             let target_y = p1.saturating_sub(1) as usize;
             super::screen::set_cursor_pos(term, term.cursor.x + 1, target_y + 1); // Pass 1-based
        }

        // --- Erasing ---
        b'J' => { // ED
            match p1_or_0 {
                0 => super::screen::erase_display_to_end(term),
                1 => super::screen::erase_display_to_start(term),
                2 => super::screen::erase_whole_display(term),
                3 => { debug!("ED 3 (Erase Scrollback) requested - Not Implemented"); }
                _ => warn!("Unknown ED parameter: {}", p1_or_0),
            }
        }
        b'K' => { // EL
            match p1_or_0 {
                0 => super::screen::erase_line_to_end(term),
                1 => super::screen::erase_line_to_start(term),
                2 => super::screen::erase_whole_line(term),
                _ => warn!("Unknown EL parameter: {}", p1_or_0),
            }
        }
        b'X' => { // ECH
             let n = p1;
             let x_start = term.cursor.x;
             let y = term.cursor.y;
             super::screen::fill_range(term, y, x_start, x_start + n as usize);
        }
        b'P' => super::screen::delete_chars(term, p1 as usize), // DCH

        // --- Scrolling ---
        b'S' => super::screen::scroll_up(term, p1 as usize),   // SU
        b'T' => super::screen::scroll_down(term, p1 as usize), // SD

        // --- Insertion/Deletion ---
        b'L' => super::screen::insert_blank_lines(term, p1 as usize), // IL
        b'M' => super::screen::delete_lines(term, p1 as usize),       // DL
        b'@' => super::screen::insert_blank_chars(term, p1 as usize), // ICH

        // --- Tabs ---
        b'I' => { // CHT
            for _ in 0..p1 { super::screen::tab(term); }
        }
        b'Z' => { // CBT
             debug!("CBT received (CSI Z) - Not Implemented");
        }
        b'g' => { // TBC
             match p1_or_0 {
                 0 => { debug!("TBC 0: Clear tab stop at current column - Not Implemented"); }
                 3 => { debug!("TBC 3: Clear all tab stops - Not Implemented"); }
                 _ => warn!("Unknown TBC parameter: {}", p1_or_0),
             }
        }

        // --- Attributes ---
        b'm' => { // SGR
            handle_sgr(term);
        }

        // --- Modes (Standard) ---
        b'h' => { warn!("Ignoring standard Set Mode (SM) sequence: CSI {} h", p1_or_0); }
        b'l' => { warn!("Ignoring standard Reset Mode (RM) sequence: CSI {} l", p1_or_0); }

        // --- Reporting ---
        b'n' => { // DSR
            match p1_or_0 {
                 5 => { debug!("DSR 5 (Status Report) received - sending OK"); /* ttywrite("\x1b[0n"); */ }
                 6 => { // CPR
                     let report = format!("\x1b[{};{}R", term.cursor.y + 1, term.cursor.x + 1);
                     debug!("DSR 6 (CPR) received - sending {}", report);
                     // TODO: Need a way to write back to PTY from Term
                 }
                 _ => warn!("Unknown DSR parameter: {}", p1_or_0),
             }
        }
        b'c' => { // DA
             if p1_or_0 == 0 {
                 debug!("DA received - sending VT100 ID (Not Implemented)");
                 // ttywrite(vtiden);
             } else {
                 warn!("Unknown DA parameter: {}", p1_or_0);
             }
        }

        // --- Scrolling Region ---
        b'r' => { // DECSTBM
            let top = get_csi_param(term, 0, 1);
            let bot = get_csi_param(term, 1, term.height as u16);
            super::screen::set_scrolling_region(term, top as usize, bot as usize);
        }

        // --- Cursor Save/Restore (ANSI.SYS / SCO) ---
        b's' => super::screen::save_cursor(term),   // SCOSC
        b'u' => super::screen::restore_cursor(term), // SCORC

        _ => { // Unhandled standard sequence
            warn!(
                "Unhandled standard CSI sequence: final='{}' (0x{:02X}), params={:?}, intermediates={:?}",
                final_byte as char, final_byte, term.csi_params, term.csi_intermediates
            );
        }
    }
    // Parameters are cleared by the caller state handler after dispatch returns
}


/// Dispatches a completed OSC sequence based on the collected string.
fn handle_osc_dispatch(term: &mut Term) {
    let osc_content = String::from_utf8_lossy(term.osc_string.as_bytes());
    trace!("Dispatching OSC: '{}'", osc_content);

    let mut parts = osc_content.splitn(2, ';');
    let code_str = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("");

    if code_str.is_empty() {
        warn!("Empty OSC code received.");
        return;
    }

    match code_str {
        "0" | "2" => { // Set window title
            debug!("OSC 0/2: Set Title to '{}' (Backend Action Needed)", arg);
        }
        "1" => { // Set icon name
             debug!("OSC 1: Set Icon Name to '{}' (Backend Action Needed)", arg);
        }
        _ => {
            warn!("Unhandled OSC sequence code: '{}', arg: '{}'", code_str, arg);
        }
    }
}

/// Handles SGR (Select Graphic Rendition) codes (CSI ... m).
fn handle_sgr(term: &mut Term) {
    let params = term.csi_params.clone();
    let nparams = params.len();
    let mut i = 0;

    if nparams == 0 {
        term.current_attributes = term.default_attributes;
        return; // Early return for empty SGR
    }

    while i < nparams {
        let p = params.get(i).copied().unwrap_or(SGR_RESET);
        trace!("Processing SGR param: {}", p);

        match p {
            SGR_RESET => term.current_attributes = term.default_attributes,
            SGR_BOLD => term.current_attributes.flags |= AttrFlags::BOLD,
            SGR_FAINT => term.current_attributes.flags |= AttrFlags::FAINT,
            SGR_ITALIC => term.current_attributes.flags |= AttrFlags::ITALIC,
            SGR_UNDERLINE => term.current_attributes.flags |= AttrFlags::UNDERLINE,
            SGR_BLINK_SLOW | SGR_BLINK_FAST => term.current_attributes.flags |= AttrFlags::BLINK,
            SGR_REVERSE => term.current_attributes.flags |= AttrFlags::REVERSE,
            SGR_HIDDEN => term.current_attributes.flags |= AttrFlags::HIDDEN,
            SGR_STRIKETHROUGH => term.current_attributes.flags |= AttrFlags::STRIKETHROUGH,
            SGR_NORMAL_INTENSITY => term.current_attributes.flags &= !(AttrFlags::BOLD | AttrFlags::FAINT),
            SGR_ITALIC_OFF => term.current_attributes.flags &= !AttrFlags::ITALIC,
            SGR_UNDERLINE_OFF => term.current_attributes.flags &= !AttrFlags::UNDERLINE,
            SGR_BLINK_OFF => term.current_attributes.flags &= !AttrFlags::BLINK,
            SGR_REVERSE_OFF => term.current_attributes.flags &= !AttrFlags::REVERSE,
            SGR_HIDDEN_OFF => term.current_attributes.flags &= !AttrFlags::HIDDEN,
            SGR_STRIKETHROUGH_OFF => term.current_attributes.flags &= !AttrFlags::STRIKETHROUGH,

            // Foreground colors
            SGR_FG_OFFSET..=37 => term.current_attributes.fg = Color::Idx((p - SGR_FG_OFFSET) as u8),
            SGR_FG_BRIGHT_OFFSET..=97 => term.current_attributes.fg = Color::Idx((p - SGR_FG_BRIGHT_OFFSET + 8) as u8),
            SGR_FG_DEFAULT => term.current_attributes.fg = term.default_attributes.fg,

            // Background colors
            SGR_BG_OFFSET..=47 => term.current_attributes.bg = Color::Idx((p - SGR_BG_OFFSET) as u8),
            SGR_BG_BRIGHT_OFFSET..=107 => term.current_attributes.bg = Color::Idx((p - SGR_BG_BRIGHT_OFFSET + 8) as u8),
            SGR_BG_DEFAULT => term.current_attributes.bg = term.default_attributes.bg,

            // Extended colors
            SGR_EXTENDED_FG | SGR_EXTENDED_BG => {
                i += 1;
                if i >= nparams { warn!("SGR: Missing specifier after {}", p); break; }
                let specifier = params.get(i).copied().unwrap_or(0);

                match specifier {
                    SGR_EXTENDED_MODE_256 => { // 256-color mode
                        i += 1;
                        if i >= nparams { warn!("SGR: Missing color index after {} ; 5", p); break; }
                        let color_idx = params.get(i).copied().unwrap_or(0);
                        if color_idx <= 255 {
                            let color = Color::Idx(color_idx as u8);
                            if p == SGR_EXTENDED_FG { term.current_attributes.fg = color; }
                            else { term.current_attributes.bg = color; }
                        } else {
                            warn!("SGR: Invalid 256-color index: {}", color_idx);
                        }
                    }
                    SGR_EXTENDED_MODE_RGB => { // Truecolor (RGB) mode
                        if i + 3 >= nparams { warn!("SGR: Missing RGB values after {} ; 2", p); break; }
                        let r = params.get(i+1).copied().unwrap_or(0).min(255) as u8;
                        let g = params.get(i+2).copied().unwrap_or(0).min(255) as u8;
                        let b = params.get(i+3).copied().unwrap_or(0).min(255) as u8;
                        i += 3;
                        let color = Color::Rgb(r, g, b);
                        if p == SGR_EXTENDED_FG { term.current_attributes.fg = color; }
                        else { term.current_attributes.bg = color; }
                    }
                    _ => {
                        warn!("SGR: Invalid specifier {} after {}", specifier, p);
                        // Attempt to recover - advance index past potential color values
                        if specifier <= 255 { i += 1; }
                        if i < nparams && params.get(i).copied().unwrap_or(0) <= 255 { i += 1; }
                        if i < nparams && params.get(i).copied().unwrap_or(0) <= 255 { i += 1; }
                        i = i.saturating_sub(1); // Decrement because outer loop increments
                    }
                }
            }

            _ => warn!("Unhandled SGR parameter: {}", p),
        }
        i += 1;
    }
}


/// Enum to represent actions needed for DEC mode setting, avoiding borrow issues.
#[derive(Debug, Clone, Copy)]
enum DecModeAction {
    SetCursorKeysAppMode(bool),
    SetOriginMode(bool),
    // SetAutowrap(bool),
    // SetReverseVideo(bool),
    SetCursorVisibility(bool),
    SetAltScreen(bool),
    SaveCursor,
    RestoreCursor,
}

/// Handles enabling DEC Private Modes (CSI ? Pn h).
fn handle_dec_mode_enable(term: &mut Term) {
    apply_dec_mode_actions(term, true);
}

/// Handles disabling DEC Private Modes (CSI ? Pn l).
fn handle_dec_mode_disable(term: &mut Term) {
    apply_dec_mode_actions(term, false);
}

/// Collects and applies DEC Private Mode actions based on parameters.
fn apply_dec_mode_actions(term: &mut Term, enable: bool) {
    let mut actions = Vec::new();
    let params = &term.csi_params; // Immutable borrow

    for &param in params {
        trace!("Collecting DEC Mode action: {} param {}", if enable { "Set" } else { "Reset" }, param);
        match param {
            DECCKM => actions.push(DecModeAction::SetCursorKeysAppMode(enable)),
            DECOM => actions.push(DecModeAction::SetOriginMode(enable)),
            DECTCEM => actions.push(DecModeAction::SetCursorVisibility(enable)),
            ALT_SCREEN_BUF_47 | ALT_SCREEN_BUF_1047 => {
                actions.push(DecModeAction::SetAltScreen(enable));
            }
            CURSOR_SAVE_RESTORE_1048 => {
                if enable { actions.push(DecModeAction::SaveCursor); }
                else { actions.push(DecModeAction::RestoreCursor); }
            }
            ALT_SCREEN_SAVE_RESTORE_1049 => {
                 if enable {
                     actions.push(DecModeAction::SaveCursor);
                     actions.push(DecModeAction::SetAltScreen(true));
                 } else {
                     actions.push(DecModeAction::SetAltScreen(false));
                     actions.push(DecModeAction::RestoreCursor);
                 }
            }
            // Ignored modes
            DECANM | DECCOLM | DECSCLM | DECARM | DECPFF | DECPEX | DECNRCM |
            MOUSE_HIGHLIGHT | MOUSE_UTF8 | MOUSE_URXVT | BLINK_CURSOR_ATT610 => {
                 trace!("Ignoring known DEC Private Mode: {}", param);
            }
            // Backend modes
            X10_MOUSE | MOUSE_BTN_REPORT | MOUSE_BTN_MOTION_REPORT | MOUSE_ANY_MOTION_REPORT |
            FOCUS_EVENT_REPORT | MOUSE_SGR_REPORT | INPUT_MODE_8BIT | BRACKETED_PASTE => {
                 debug!("Ignoring backend-handled DEC mode {}", param);
            }
            _ => warn!("Unhandled DEC Private Mode parameter: {}", param),
        }
    }

    // Apply collected actions mutably
    for action in actions {
        match action {
            DecModeAction::SetCursorKeysAppMode(val) => term.dec_modes.cursor_keys_app_mode = val,
            DecModeAction::SetOriginMode(val) => {
                if val { super::screen::enable_origin_mode(term); }
                else { super::screen::disable_origin_mode(term); }
            }
            DecModeAction::SetCursorVisibility(visible) => {
                 debug!("DECTCEM Action: Visible={} (Backend Action Needed)", visible);
            }
            DecModeAction::SetAltScreen(enter) => {
                if enter { super::screen::enter_alt_screen(term); }
                else { super::screen::exit_alt_screen(term); }
            }
            DecModeAction::SaveCursor => super::screen::save_cursor(term),
            DecModeAction::RestoreCursor => super::screen::restore_cursor(term),
        }
    }
}

#[cfg(test)]
mod tests;
