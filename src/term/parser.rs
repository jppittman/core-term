// src/term/parser.rs

//! Handles parsing of input byte streams, including escape sequences (CSI, OSC).

// Note: `Term` methods related to screen manipulation (scrolling, erasing, cursor movement, etc.)
// will be called from here but defined in `screen.rs`.
use super::{Term, ParserState}; // Import Term and ParserState from parent module (mod.rs)
use crate::glyph::{Color, AttrFlags, REPLACEMENT_CHARACTER}; // Import glyph types
use super::{MAX_CSI_PARAMS, MAX_CSI_INTERMEDIATES, MAX_OSC_STRING_LEN}; // Import constants
use std::cmp::min;

// --- Helper Functions ---

/// Determines expected UTF-8 character length from the first byte.
/// Returns 1 for invalid start bytes to allow consuming them.
pub(super) fn utf8_char_len(byte: u8) -> usize { // Make pub(super) or pub based on need
    match byte {
        0x00..=0x7F => 1,
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => 1,
    }
}

// --- Parser State Handling Functions (taking `&mut Term`) ---

/// Clears CSI parameters and intermediates.
pub(super) fn clear_csi_params(term: &mut Term) {
    term.csi_params.clear();
    term.csi_intermediates.clear();
}

/// Clears OSC string buffer.
pub(super) fn clear_osc_string(term: &mut Term) {
    term.osc_string.clear();
}

/// Appends a digit to the last CSI parameter being built.
pub(super) fn push_csi_param(term: &mut Term, digit: u16) {
    if term.csi_params.len() > MAX_CSI_PARAMS {
         return;
    }
    let current_param = if term.csi_params.is_empty() {
        term.csi_params.push(0);
        term.csi_params.last_mut().unwrap()
    } else {
        term.csi_params.last_mut().unwrap()
    };
    *current_param = current_param.saturating_mul(10).saturating_add(digit);
}

/// Moves to the next CSI parameter slot (appends a 0).
pub(super) fn next_csi_param(term: &mut Term) {
     if term.csi_params.is_empty() {
         term.csi_params.push(0);
     }
     if term.csi_params.len() < MAX_CSI_PARAMS {
         term.csi_params.push(0);
     }
}

 /// Gets the nth CSI parameter (0-based index), defaulting if absent/0.
 pub(super) fn get_csi_param(term: &Term, index: usize, default: u16) -> u16 {
    match term.csi_params.get(index) {
        Some(&p) if p > 0 => p,
        _ => default,
    }
}

 /// Gets the nth CSI parameter (0-based index), returning 0 if absent.
 pub(super) fn get_csi_param_or_0(term: &Term, index: usize) -> u16 {
    *term.csi_params.get(index).unwrap_or(&0)
}


// --- State Transition Handlers ---

/// Handles bytes in the Ground state (normal operation).
pub(super) fn handle_ground_byte(term: &mut Term, byte: u8) {
    match byte {
        // C0 Control Codes - Delegate to screen module for actions
        0x08 => super::screen::backspace(term),
        0x09 => super::screen::tab(term),
        0x0A..=0x0C => super::screen::newline(term), // LF, VT, FF
        0x0D => super::screen::carriage_return(term),
        0x00..=0x07 | 0x0E..=0x1A | 0x1C..=0x1F => { /* Ignore other C0 */ }
        0x1B => { // ESC
            term.parser_state = ParserState::Escape;
            clear_csi_params(term);
            clear_osc_string(term);
        }
        // Printable characters (including UTF-8 sequences)
        _ => {
            match term.utf8_decoder.decode(byte) {
                Ok(Some(c)) => super::screen::handle_printable(term, c), // Delegate to screen module
                Ok(None) => { /* Need more bytes */ }
                Err(_) => {
                    super::screen::handle_printable(term, REPLACEMENT_CHARACTER);
                }
            }
        }
    }
}

/// Handles bytes after an ESC (0x1B) has been received.
pub(super) fn handle_escape_byte(term: &mut Term, byte: u8) {
    match byte {
        b'[' => term.parser_state = ParserState::CSIEntry,
        b']' => term.parser_state = ParserState::OSCParam,
        b'c' => { // RIS - Reset to Initial State
            super::screen::reset(term); // Delegate reset logic
            // TODO: Inform backend about title reset if necessary
            term.parser_state = ParserState::Ground;
        }
        b'7' => { // DECSC - Save Cursor
            super::screen::save_cursor(term); // Delegate
            term.parser_state = ParserState::Ground;
        }
        b'8' => { // DECRC - Restore Cursor
            super::screen::restore_cursor(term); // Delegate
            term.parser_state = ParserState::Ground;
        }
        // Ignore Shift/Character Set selections for now
        b'(' | b')' | b'*' | b'+' => term.parser_state = ParserState::Ground, // Ignore SCS
        // Ignore keypad modes (handled by CSI ? 1 h/l)
        b'=' | b'>' => term.parser_state = ParserState::Ground,
        // Unhandled ESC sequences return to Ground state
        _ => term.parser_state = ParserState::Ground,
    }
}

/// Handles the first byte after ESC [.
pub(super) fn handle_csi_entry_byte(term: &mut Term, byte: u8) {
    match byte {
        b'0'..=b'9' => {
            push_csi_param(term, (byte - b'0') as u16);
            term.parser_state = ParserState::CSIParam;
        }
        b';' => {
            next_csi_param(term);
            term.parser_state = ParserState::CSIParam;
        }
        b'?' | b'>' | b'!' | b':' => {
            if term.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                 term.csi_intermediates.push(byte as char);
                 term.parser_state = ParserState::CSIIntermediate;
             } else {
                 term.parser_state = ParserState::CSIIgnore;
             }
        }
        0x40..=0x7E => {
            csi_dispatch(term, byte);
            term.parser_state = ParserState::Ground;
        }
        _ => term.parser_state = ParserState::Ground,
    }
}

/// Handles CSI parameter bytes (digits, ';').
pub(super) fn handle_csi_param_byte(term: &mut Term, byte: u8) {
    match byte {
        b'0'..=b'9' => push_csi_param(term, (byte - b'0') as u16),
        b';' => next_csi_param(term),
        0x20..=0x2F => { // Intermediate bytes
             if term.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                 term.csi_intermediates.push(byte as char);
                 term.parser_state = ParserState::CSIIntermediate;
             } else {
                 term.parser_state = ParserState::CSIIgnore;
             }
        }
        0x40..=0x7E => { // Final byte
            csi_dispatch(term, byte);
            term.parser_state = ParserState::Ground;
        }
        _ => term.parser_state = ParserState::Ground,
    }
}

/// Handles CSI intermediate bytes (after params, before final byte).
pub(super) fn handle_csi_intermediate_byte(term: &mut Term, byte: u8) {
    match byte {
        0x20..=0x2F => { // Collect intermediate byte
            if term.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                term.csi_intermediates.push(byte as char);
            } else {
                term.parser_state = ParserState::CSIIgnore;
            }
        }
        0x40..=0x7E => { // Final byte
            csi_dispatch(term, byte);
            term.parser_state = ParserState::Ground;
        }
        b'0'..=b'9' => { // Parameters after intermediates
            push_csi_param(term, (byte - b'0') as u16);
            term.parser_state = ParserState::CSIParam;
        }
        b';' => { // Parameter separator after intermediates
            next_csi_param(term);
            term.parser_state = ParserState::CSIParam;
        }
        _ => term.parser_state = ParserState::Ground,
    }
}

/// Handles bytes when ignoring the rest of a CSI sequence.
pub(super) fn handle_csi_ignore_byte(term: &mut Term, byte: u8) {
    if (0x40..=0x7E).contains(&byte) {
        term.parser_state = ParserState::Ground;
    }
}

/// Handles the parameter part of an OSC sequence (e.g., '0;')
pub(super) fn handle_osc_param_byte(term: &mut Term, byte: u8) {
    match byte {
        b'0'..=b'9' | b';' => {
            if term.osc_string.len() < MAX_OSC_STRING_LEN {
                term.osc_string.push(byte as char);
            } else {
                term.parser_state = ParserState::Ground;
            }
            if byte == b';' {
                term.parser_state = ParserState::OSCParse;
            }
        }
        0x07 => { // BEL
            handle_osc_end(term, true);
            term.parser_state = ParserState::Ground;
        }
        0x1B => { // ESC potentially starts ST
            term.parser_state = ParserState::OSCParse;
            handle_osc_parse_byte(term, byte);
        }
        _ => { // Assume start of content if not param, BEL, or ESC
            term.parser_state = ParserState::OSCParse;
            handle_osc_parse_byte(term, byte);
        }
    }
}

/// Handles bytes during the main OSC string parsing (after params, before ST/BEL).
pub(super) fn handle_osc_parse_byte(term: &mut Term, byte: u8) {
    match byte {
        0x07 => { // BEL terminates OSC sequence
            handle_osc_end(term, true);
            term.parser_state = ParserState::Ground;
        }
        0x1B => { // Check for ESC \ (ST - String Terminator)
             // TODO: Implement proper ST (ESC \) handling
             handle_osc_end(term, false);
             term.parser_state = ParserState::Ground;
        }
        0x00..=0x06 | 0x08..=0x1A | 0x1C..=0x1F => {} // Ignore C0
        _ => {
            if term.osc_string.len() < MAX_OSC_STRING_LEN {
                term.osc_string.push(byte as char);
            }
        }
    }
}

/// Handles the end of an OSC sequence (called by BEL or ST).
fn handle_osc_end(term: &mut Term, _is_bel: bool) { // Prefix unused _is_bel
    if let Some((param_str, content)) = term.osc_string.split_once(';') {
        match param_str {
            "0" | "2" => {
                // TODO: Inform the backend/UI to set the title
                println!("OSC: Set Title: {}", content); // Placeholder
            }
            _ => {} // Unknown OSC parameter
        }
    }
    clear_osc_string(term);
}

/// Dispatches action based on the final byte of a CSI sequence.
fn csi_dispatch(term: &mut Term, final_byte: u8) {
    let is_private = term.csi_intermediates.first() == Some(&'?');
    let top = 0; // TODO: Use term.top when scrolling region implemented
    let bot = term.height - 1; // TODO: Use term.bot when scrolling region implemented

    match final_byte {
        // Cursor Movement
        b'A' => { let n = get_csi_param(term, 0, 1) as isize; super::screen::move_cursor(term, 0, -n); } // CUU
        b'B' => { let n = get_csi_param(term, 0, 1) as isize; super::screen::move_cursor(term, 0, n); }  // CUD
        b'C' => { let n = get_csi_param(term, 0, 1) as isize; super::screen::move_cursor(term, n, 0); }  // CUF
        b'D' => { let n = get_csi_param(term, 0, 1) as isize; super::screen::move_cursor(term, -n, 0); } // CUB
        b'E' => { let n = get_csi_param(term, 0, 1) as isize; super::screen::move_cursor(term, 0, n); term.cursor.x = 0; } // CNL
        b'F' => { let n = get_csi_param(term, 0, 1) as isize; super::screen::move_cursor(term, 0, -n); term.cursor.x = 0; } // CPL
        b'G' => { let n = get_csi_param(term, 0, 1); term.cursor.x = min(n.saturating_sub(1) as usize, term.width.saturating_sub(1)); } // CHA
        b'H' | b'f' => { // CUP / HVP
            let y = get_csi_param(term, 0, 1);
            let x = get_csi_param(term, 1, 1);
            super::screen::set_cursor_pos(term, x.saturating_sub(1) as usize, y.saturating_sub(1) as usize);
        }

        // Erasing
        b'J' => { // ED
            match get_csi_param_or_0(term, 0) {
                0 => super::screen::erase_display_to_end(term),
                1 => super::screen::erase_display_to_start(term),
                2 => super::screen::erase_whole_display(term),
                _ => {}
            }
        }
        b'K' => { // EL
            match get_csi_param_or_0(term, 0) {
                0 => super::screen::erase_line_to_end(term),
                1 => super::screen::erase_line_to_start(term),
                2 => super::screen::erase_whole_line(term),
                _ => {}
            }
        }

        // Attributes
        b'm' => handle_sgr(term), // SGR

        // Insert/Delete Characters/Lines
        b'@' => { let n = get_csi_param(term, 0, 1); super::screen::insert_blank_chars(term, n as usize); } // ICH
        b'L' => { let n = get_csi_param(term, 0, 1); super::screen::insert_blank_lines(term, n as usize); } // IL
        b'P' => { let n = get_csi_param(term, 0, 1); super::screen::delete_chars(term, n as usize); } // DCH
        b'M' => { let n = get_csi_param(term, 0, 1); super::screen::delete_lines(term, n as usize); } // DL

        // Cursor Save/Restore (using ANSI standard codes)
        b's' => super::screen::save_cursor(term), // SCOSC
        b'u' => super::screen::restore_cursor(term), // SCORC


        // Mode Setting
        b'h' if is_private => handle_dec_mode_set(term, true),  // DECSET
        b'l' if is_private => handle_dec_mode_set(term, false), // DECRST

        // Special handling for Alt Screen Buffer (DECSET/DECRST 1049)
        b'h' if is_private && get_csi_param_or_0(term, 0) == 1049 => {
             super::screen::enter_alt_screen(term);
         }
         b'l' if is_private && get_csi_param_or_0(term, 0) == 1049 => {
             super::screen::exit_alt_screen(term);
         }

        // Scrolling
        b'S' => { let n = get_csi_param(term, 0, 1); super::screen::scroll_up(term, top, bot, n as usize); } // SU
        b'T' => { let n = get_csi_param(term, 0, 1); super::screen::scroll_down(term, top, bot, n as usize); } // SD

        // Set Scrolling Region
        b'r' if !is_private => { // DECSTBM
             let _top = get_csi_param(term, 0, 1).saturating_sub(1) as usize; // Prefixed unused
             let _bottom = get_csi_param(term, 1, term.height as u16).saturating_sub(1) as usize; // Prefixed unused
             // TODO: Implement tsetscroll(top, bottom) and ensure cursor stays within region
             super::screen::set_cursor_pos(term, 0, 0); // Move home after setting region
         }

        _ => { /* Ignore unknown CSI sequences */ }
    }
}

/// Handles SGR (Select Graphic Rendition) codes (CSI ... m).
fn handle_sgr(term: &mut Term) {
    if term.csi_params.is_empty() || get_csi_param_or_0(term, 0) == 0 {
        term.current_attributes = term.default_attributes;
        if term.csi_params.len() > 1 {
            process_sgr_codes(term, 1);
        }
        return;
    }
    process_sgr_codes(term, 0);
}

/// Processes SGR codes starting from a given index in `csi_params`.
fn process_sgr_codes(term: &mut Term, start_index: usize) {
    let mut i = start_index;
    // Clone params to avoid borrow checker issues with get_csi_param_or_0 inside loop
    let params = term.csi_params.clone();

    while i < params.len() {
        // Use the cloned params for reading
        let code = *params.get(i).unwrap_or(&0);
        match code {
            0 => term.current_attributes = term.default_attributes,
            1 => term.current_attributes.flags |= AttrFlags::BOLD,
            3 => term.current_attributes.flags |= AttrFlags::ITALIC,
            4 => term.current_attributes.flags |= AttrFlags::UNDERLINE,
            7 => term.current_attributes.flags |= AttrFlags::REVERSE,
            8 => term.current_attributes.flags |= AttrFlags::HIDDEN,
            9 => term.current_attributes.flags |= AttrFlags::STRIKETHROUGH,
            22 => term.current_attributes.flags &= !AttrFlags::BOLD,
            23 => term.current_attributes.flags &= !AttrFlags::ITALIC,
            24 => term.current_attributes.flags &= !AttrFlags::UNDERLINE,
            27 => term.current_attributes.flags &= !AttrFlags::REVERSE,
            28 => term.current_attributes.flags &= !AttrFlags::HIDDEN,
            29 => term.current_attributes.flags &= !AttrFlags::STRIKETHROUGH,
            30..=37 => term.current_attributes.fg = Color::Idx((code - 30) as u8),
            38 => { // Extended FG
                i += 1;
                match params.get(i).unwrap_or(&0) {
                    5 => { // 256-color
                        i += 1;
                        let idx = *params.get(i).unwrap_or(&0) as u8;
                        term.current_attributes.fg = Color::Idx(idx);
                    }
                    2 => { // Truecolor
                        if i + 3 < params.len() {
                            let r = *params.get(i + 1).unwrap_or(&0) as u8;
                            let g = *params.get(i + 2).unwrap_or(&0) as u8;
                            let b = *params.get(i + 3).unwrap_or(&0) as u8;
                            term.current_attributes.fg = Color::Rgb(r, g, b);
                            i += 3;
                        } else {
                            i = params.len(); // Error
                        }
                    }
                    _ => { i += 1; } // Skip potential value
                }
            }
            39 => term.current_attributes.fg = term.default_attributes.fg,
            40..=47 => term.current_attributes.bg = Color::Idx((code - 40) as u8),
            48 => { // Extended BG
                i += 1;
                match params.get(i).unwrap_or(&0) {
                    5 => { // 256-color
                        i += 1;
                        let idx = *params.get(i).unwrap_or(&0) as u8;
                        term.current_attributes.bg = Color::Idx(idx);
                    }
                    2 => { // Truecolor
                        if i + 3 < params.len() {
                            let r = *params.get(i + 1).unwrap_or(&0) as u8;
                            let g = *params.get(i + 2).unwrap_or(&0) as u8;
                            let b = *params.get(i + 3).unwrap_or(&0) as u8;
                            term.current_attributes.bg = Color::Rgb(r, g, b);
                            i += 3;
                        } else {
                            i = params.len(); // Error
                        }
                    }
                     _ => { i += 1; } // Skip potential value
                }
            }
            49 => term.current_attributes.bg = term.default_attributes.bg,
            90..=97 => term.current_attributes.fg = Color::Idx((code - 90 + 8) as u8),
            100..=107 => term.current_attributes.bg = Color::Idx((code - 100 + 8) as u8),
            _ => { /* Ignore unknown SGR codes */ }
        }
        i += 1;
    }
}


/// Handles DEC Private Mode Set/Reset sequences (CSI ? Pn h/l).
fn handle_dec_mode_set(term: &mut Term, enable: bool) {
    for i in 0..term.csi_params.len() {
         let mode = get_csi_param_or_0(term, i);
         match mode {
            1 => term.dec_modes.cursor_keys_app_mode = enable, // DECCKM
            1049 => { /* Handled specially in csi_dispatch */ }
            _ => { /* Ignore unknown DEC private modes */ }
        }
    }
}
