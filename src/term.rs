// src/term.rs
// Defines the terminal state (Term) and its ANSI processing logic.

use crate::glyph::{Glyph, Attributes, AttrFlags, ColorSpec, Color};

// ** FIX: Correct libc imports and use std RawFd **
use libc::{ioctl, pid_t, winsize, TIOCSWINSZ};
use std::os::unix::io::{RawFd, AsRawFd}; // Use std RawFd
use std::fs::File;
use std::io;
// ** FIX: Removed unused std::mem **
// use std::mem;
// ** FIX: Removed unused anyhow import **
use anyhow::{Context, Result}; // Keep Result, Context

// Constants used only within Term logic
const ESC_ARG_SIZ: usize = 16;

// --- Structs and Enums ---

// Basic struct to track common DEC private modes (kept private to this module)
#[derive(Debug, Default, Clone, Copy)]
struct DecPrivateModes {
    cursor_keys_app_mode: bool,
    alt_screen_buffer: bool,
    bracketed_paste: bool,
    mouse_reporting_btn: bool,
    mouse_reporting_motion: bool,
    mouse_reporting_all: bool,
    mouse_reporting_focus: bool,
    mouse_reporting_sgr: bool,
}

// ANSI Parser States (kept private)
#[derive(Debug, PartialEq, Clone, Copy)]
enum ParserState {
    Ground, Escape, Csi, Osc, EscIntermediate,
}

// Enum for erase directions (kept private)
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum EraseDirection { ToEnd, ToStart, WholeLine, Unsupported(usize) }

// Enum for parsed CSI sequences (kept private)
#[derive(Debug, PartialEq)]
enum CsiSequence {
    CursorPosition { row: usize, col: usize },
    EraseInLine(EraseDirection),
    EraseInDisplay(EraseDirection),
    CursorUp { n: usize },
    CursorDown { n: usize },
    CursorForward { n: usize },
    CursorBackward { n: usize },
    PrivateModeSet(usize),
    PrivateModeReset(usize),
    Sgr { params: Vec<usize> },
    Unsupported(u8, Vec<usize>),
}


// Simplified Terminal State (Make the struct and relevant fields/methods public)
#[derive(Debug)]
pub struct Term { // Made pub
    _child_pid: pid_t, // Keep private for now
    pub pty_parent: std::fs::File, // Made pub for handle_pty_read/handle_key_press in main/backends
    pub screen: Vec<Vec<Glyph>>, // Made pub for drawing
    pub cols: usize, // Made pub for drawing/resize checks
    pub rows: usize, // Made pub for drawing/resize checks
    pub cursor_x: usize, // Made pub for drawing
    pub cursor_y: usize, // Made pub for drawing
    parser_state: ParserState, // Keep private
    escape_buffer: Vec<u8>, // Keep private
    dec_modes: DecPrivateModes, // Keep private
    current_attributes: Attributes, // Keep private (exposed via Glyphs on screen)
}

impl Term {
    // Initialize a new terminal state (Make public)
    pub fn new(child_pid: pid_t, pty_parent: File, cols: usize, rows: usize) -> Self {
        let cols = cols.max(1);
        let rows = rows.max(1);
        let mut screen = Vec::with_capacity(rows);
        for _ in 0..rows { screen.push(vec![Glyph::default(); cols]); }
        Term {
            _child_pid: child_pid, pty_parent, screen, cols, rows,
            cursor_x: 0, cursor_y: 0, parser_state: ParserState::Ground,
            escape_buffer: Vec::new(), dec_modes: DecPrivateModes::default(),
            current_attributes: Attributes::default(),
        }
    }

    // Resize the terminal (Make public)
    pub fn resize(&mut self, new_cols: usize, new_rows: usize) -> Result<()> {
        if new_cols == 0 || new_rows == 0 { anyhow::bail!("Terminal resize dimensions must be positive (got {}x{})", new_cols, new_rows); }
        if self.cols == new_cols && self.rows == new_rows { return Ok(()); }
        let old_cols = self.cols;
        self.screen.resize_with(new_rows, || vec![Glyph::default(); old_cols]);
        for row in self.screen.iter_mut() { row.resize(new_cols, Glyph::default()); }
        self.cols = new_cols;
        self.rows = new_rows;
        self.cursor_x = std::cmp::min(self.cursor_x, self.cols.saturating_sub(1));
        self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));
        let mut win_size = winsize { ws_row: self.rows as u16, ws_col: self.cols as u16, ws_xpixel: 0, ws_ypixel: 0 };
        let fd = self.pty_parent.as_raw_fd();
        if fd >= 0 {
            // SAFETY: ioctl is FFI.
            unsafe { if ioctl(fd, TIOCSWINSZ, &mut win_size) < 0 && cfg!(not(test)) { return Err(io::Error::last_os_error()).context("TIOCSWINSZ failed"); } }
        }
        Ok(())
    }

    // Process a single byte from the PTY based on parser state (Keep private)
    fn process_byte(&mut self, byte: u8) {
        match self.parser_state {
            ParserState::Ground => match byte {
                0x1B => { self.parser_state = ParserState::Escape; self.escape_buffer.clear(); }
                b'\n' => { self.cursor_y = std::cmp::min(self.cursor_y + 1, self.rows - 1); }
                b'\r' => { self.cursor_x = 0; }
                0x08 => { self.cursor_x = self.cursor_x.saturating_sub(1); }
                0x07 => { /* BEL */ }
                0..=6 | 9..=12 | 14..=31 | 127 => { /* Ignore */ }
                _ => { // Printable
                    self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));
                    if self.cursor_x >= self.cols { self.cursor_x = 0; self.cursor_y = std::cmp::min(self.cursor_y + 1, self.rows - 1); }
                    self.cursor_x = std::cmp::min(self.cursor_x, self.cols.saturating_sub(1));
                    self.cursor_y = std::cmp::min(self.cursor_y, self.rows.saturating_sub(1));
                    if self.cursor_y < self.screen.len() && self.cursor_x < self.screen[self.cursor_y].len() {
                        self.screen[self.cursor_y][self.cursor_x] = Glyph { c: byte as char, attr: self.current_attributes };
                    }
                    self.cursor_x += 1;
                }
            },
            ParserState::Escape => match byte {
                b'[' => { self.parser_state = ParserState::Csi; self.escape_buffer.clear(); }
                b']' => { self.parser_state = ParserState::Osc; self.escape_buffer.clear(); }
                b'(' | b')' | b'*' | b'+' => { self.escape_buffer.push(byte); self.parser_state = ParserState::EscIntermediate; }
                b'=' => { self.parser_state = ParserState::Ground; } // DECKPAM
                b'>' => { self.parser_state = ParserState::Ground; } // DECKPNM
                _ => { self.parser_state = ParserState::Ground; }
            },
            ParserState::EscIntermediate => {
                let _intermediate = self.escape_buffer.pop().unwrap_or(0);
                self.parser_state = ParserState::Ground;
            }
            ParserState::Csi => {
                self.escape_buffer.push(byte);
                if byte >= 0x40 && byte <= 0x7E { // Final byte
                    match self.parse_csi_sequence() {
                        Ok(sequence) => self.handle_csi_sequence(sequence),
                        Err(e) => eprintln!("Error parsing CSI: {}", e),
                    }
                    self.parser_state = ParserState::Ground; self.escape_buffer.clear();
                } else if byte < 0x20 || byte > 0x3F { // Invalid byte
                    self.parser_state = ParserState::Ground; self.escape_buffer.clear();
                }
            }
            ParserState::Osc => match byte {
                0x07 => { self.parser_state = ParserState::Ground; self.escape_buffer.clear(); }
                0x1B => { self.escape_buffer.push(byte); }
                b'\\' if self.escape_buffer.last() == Some(&0x1B) => { self.escape_buffer.pop(); self.parser_state = ParserState::Ground; self.escape_buffer.clear(); }
                _ => { self.escape_buffer.push(byte); }
            },
        }
    }

    // Parse a completed CSI sequence into a CsiSequence enum (Keep private)
    fn parse_csi_sequence(&self) -> Result<CsiSequence> {
        let sequence = &self.escape_buffer;
        if sequence.is_empty() { anyhow::bail!("Empty CSI sequence buffer"); }
        let final_byte = *sequence.last().unwrap();
        let params_bytes = &sequence[..sequence.len() - 1];
        let mut cursor = 0;
        let mut is_private = false;
        if params_bytes.get(cursor) == Some(&b'?') { is_private = true; cursor += 1; }
        let mut params: Vec<usize> = Vec::new();
        let mut current_param_start = cursor;
        while cursor < params_bytes.len() {
            if params_bytes[cursor] == b';' {
                 let slice = &params_bytes[current_param_start..cursor];
                 params.push(if slice.is_empty() { 0 } else { String::from_utf8_lossy(slice).parse().unwrap_or(0) });
                 current_param_start = cursor + 1;
                 if params.len() >= ESC_ARG_SIZ { break; }
             }
            cursor += 1;
        }
        if current_param_start <= params_bytes.len() {
             let slice = &params_bytes[current_param_start..params_bytes.len()];
             params.push(if slice.is_empty() { 0 } else { String::from_utf8_lossy(slice).parse().unwrap_or(0) });
        }
        let get_param = |idx: usize, default: usize| -> usize { match params.get(idx) { Some(&0) => 0, Some(&v) => v, None => default } };
        let parse_erase = |idx: usize| match get_param(idx, 0) { 0=>EraseDirection::ToEnd, 1=>EraseDirection::ToStart, 2=>EraseDirection::WholeLine, v=>EraseDirection::Unsupported(v) };
        match final_byte {
            b'H'|b'f' => Ok(CsiSequence::CursorPosition{ row: get_param(0,1), col: get_param(1,1) }),
            b'A' => Ok(CsiSequence::CursorUp{ n: get_param(0,1) }),
            b'B' => Ok(CsiSequence::CursorDown{ n: get_param(0,1) }),
            b'C' => Ok(CsiSequence::CursorForward{ n: get_param(0,1) }),
            b'D' => Ok(CsiSequence::CursorBackward{ n: get_param(0,1) }),
            b'J' => Ok(CsiSequence::EraseInDisplay(parse_erase(0))),
            b'K' => Ok(CsiSequence::EraseInLine(parse_erase(0))),
            b'h' if is_private => Ok(CsiSequence::PrivateModeSet(get_param(0,0))),
            b'l' if is_private => Ok(CsiSequence::PrivateModeReset(get_param(0,0))),
            b'm' => Ok(CsiSequence::Sgr { params }),
            _ => Ok(CsiSequence::Unsupported(final_byte, params)),
        }
    }


    // Handle a parsed CSI sequence, updating terminal state (Keep private)
    fn handle_csi_sequence(&mut self, sequence: CsiSequence) {
        match sequence {
            CsiSequence::CursorPosition { row, col } => {
                self.cursor_y = row.saturating_sub(1).min(self.rows.saturating_sub(1));
                self.cursor_x = col.saturating_sub(1).min(self.cols.saturating_sub(1));
            }
            CsiSequence::EraseInLine(direction) => {
                if self.cursor_y >= self.rows { return; }
                match direction {
                    EraseDirection::ToEnd => { for x in self.cursor_x..self.cols { self.screen[self.cursor_y][x] = Glyph::default(); } }
                    EraseDirection::ToStart => { let end = self.cursor_x.min(self.cols.saturating_sub(1)); for x in 0..=end { self.screen[self.cursor_y][x] = Glyph::default(); } }
                    EraseDirection::WholeLine => { self.screen[self.cursor_y].fill(Glyph::default()); }
                    EraseDirection::Unsupported(_) => {}
                }
            }
             CsiSequence::EraseInDisplay(direction) => {
                 match direction {
                     EraseDirection::ToEnd => {
                         if self.cursor_y < self.rows { for x in self.cursor_x..self.cols { self.screen[self.cursor_y][x] = Glyph::default(); } }
                         for y in self.cursor_y.saturating_add(1)..self.rows { self.screen[y].fill(Glyph::default()); }
                     }
                     EraseDirection::ToStart => {
                          if self.cursor_y < self.rows { let end = self.cursor_x.min(self.cols.saturating_sub(1)); for x in 0..=end { self.screen[self.cursor_y][x] = Glyph::default(); } }
                         for y in 0..self.cursor_y { if y < self.rows { self.screen[y].fill(Glyph::default()); } }
                     }
                     EraseDirection::WholeLine => { for y in 0..self.rows { if y < self.rows { self.screen[y].fill(Glyph::default()); } } }
                    EraseDirection::Unsupported(_) => {}
                 }
             }
            CsiSequence::CursorUp { n } => { let n = n.max(1); self.cursor_y = self.cursor_y.saturating_sub(n).max(0); }
            CsiSequence::CursorDown { n } => { let n = n.max(1); self.cursor_y = self.cursor_y.saturating_add(n).min(self.rows.saturating_sub(1)); }
            CsiSequence::CursorForward { n } => { let n = n.max(1); self.cursor_x = self.cursor_x.saturating_add(n).min(self.cols.saturating_sub(1)); }
            CsiSequence::CursorBackward { n } => { let n = n.max(1); self.cursor_x = self.cursor_x.saturating_sub(n).max(0); }
            CsiSequence::PrivateModeSet(mode) => self.handle_dec_mode_set(mode, true),
            CsiSequence::PrivateModeReset(mode) => self.handle_dec_mode_set(mode, false),
            CsiSequence::Sgr { params } => self.handle_sgr(params),
            CsiSequence::Unsupported(_, _) => { /* Silently ignore */ }
        }
    }

    // Handles SGR (Select Graphic Rendition) sequences (CSI m) (Keep private)
    fn handle_sgr(&mut self, params: Vec<usize>) {
        if params.is_empty() { self.current_attributes = Attributes::default(); return; }
        let mut i = 0;
        while i < params.len() {
            let param = params[i] as u8;
            match param {
                0 => { self.current_attributes = Attributes::default(); }
                1 => { self.current_attributes.flags |= AttrFlags::BOLD; }
                4 => { self.current_attributes.flags |= AttrFlags::UNDERLINE; }
                7 => { self.current_attributes.flags |= AttrFlags::REVERSE; }
                22 => { self.current_attributes.flags &= !AttrFlags::BOLD; }
                24 => { self.current_attributes.flags &= !AttrFlags::UNDERLINE; }
                27 => { self.current_attributes.flags &= !AttrFlags::REVERSE; }
                30..=37 => { if let Some(c) = Color::from_index(param - 30) { self.current_attributes.fg = ColorSpec::Idx(c.index()); } }
                40..=47 => { if let Some(c) = Color::from_index(param - 40) { self.current_attributes.bg = ColorSpec::Idx(c.index()); } }
                38 => { if i + 1 < params.len() { match params[i + 1] as u8 { 5 if i + 2 < params.len() => { self.current_attributes.fg = ColorSpec::Idx(params[i + 2] as u8); i += 2; } 2 if i + 4 < params.len() => { let r = params[i + 2].min(255) as u8; let g = params[i + 3].min(255) as u8; let b = params[i + 4].min(255) as u8; self.current_attributes.fg = ColorSpec::Rgb(r, g, b); i += 4; } _ => break } } else { break; } }
                48 => { if i + 1 < params.len() { match params[i + 1] as u8 { 5 if i + 2 < params.len() => { self.current_attributes.bg = ColorSpec::Idx(params[i + 2] as u8); i += 2; } 2 if i + 4 < params.len() => { let r = params[i + 2].min(255) as u8; let g = params[i + 3].min(255) as u8; let b = params[i + 4].min(255) as u8; self.current_attributes.bg = ColorSpec::Rgb(r, g, b); i += 4; } _ => break } } else { break; } }
                39 => { self.current_attributes.fg = ColorSpec::Default; }
                49 => { self.current_attributes.bg = ColorSpec::Default; }
                90..=97 => { if let Some(c) = Color::from_index(param - 90 + 8) { self.current_attributes.fg = ColorSpec::Idx(c.index()); } }
                100..=107 => { if let Some(c) = Color::from_index(param - 100 + 8) { self.current_attributes.bg = ColorSpec::Idx(c.index()); } }
                _ => { /* Ignore */ }
            }
            i += 1;
        }
    }

    // Helper to handle DEC Private Mode setting/resetting (Keep private)
    fn handle_dec_mode_set(&mut self, mode: usize, set: bool) {
        match mode {
            1 => self.dec_modes.cursor_keys_app_mode = set,
            12 | 25 => { /* Ignore */ }
            47 | 1047 | 1049 => { if set != self.dec_modes.alt_screen_buffer { self.dec_modes.alt_screen_buffer = set; /* TODO: Swap buffers */ } }
            1000 => self.dec_modes.mouse_reporting_btn = set,
            1002 => self.dec_modes.mouse_reporting_motion = set,
            1003 => self.dec_modes.mouse_reporting_all = set,
            1004 => self.dec_modes.mouse_reporting_focus = set,
            1005 => { /* Ignore */ }
            1006 => self.dec_modes.mouse_reporting_sgr = set,
            2004 => self.dec_modes.bracketed_paste = set,
            7727 => { /* Ignore */ }
            _ => { /* Ignore */ }
        }
    }

    // Process incoming data from the PTY based on parser state (Make public)
    pub fn process_pty_data(&mut self, data: &[u8]) -> Result<()> {
        for &byte in data { self.process_byte(byte); }
        Ok(())
    }

    // Get the raw file descriptor for the PTY parent (Keep public?)
    // Only needed if handle_key_press writes directly, which it shouldn't
    #[allow(dead_code)]
    pub fn pty_parent_fd(&self) -> RawFd { self.pty_parent.as_raw_fd() }
}


// --- Unit Tests for Term ---
#[cfg(test)]
mod tests {
    use super::*; // Import items from outer module (Term, etc.)
    use crate::glyph::{ColorSpec, Color, AttrFlags, Attributes, Glyph}; // Import glyph items needed for tests
    use std::fs::OpenOptions;

    // Helper to create a dummy Term instance for testing parser logic
    fn create_test_term(cols: usize, rows: usize) -> Term {
        let dummy_file = OpenOptions::new().read(true).write(true).open("/dev/null").expect("Failed to open /dev/null for test");
        Term::new(0, dummy_file, cols, rows)
    }

    // Helper to process a string of bytes through the term parser
    fn process_str(term: &mut Term, input: &str) {
        let _ = term.process_pty_data(input.as_bytes());
    }

    #[test]
    fn test_printable_chars() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abc");

        assert_eq!(term.cursor_x, 3);
        assert_eq!(term.cursor_y, 0);
        assert_eq!(term.screen[0][0].c, 'a');
        assert_eq!(term.screen[0][1].c, 'b');
        assert_eq!(term.screen[0][2].c, 'c');
        assert_eq!(term.screen[0][3].c, ' ');
    }

    #[test]
    fn test_line_wrap() {
        let mut term = create_test_term(3, 2);
        process_str(&mut term, "abcd");

        assert_eq!(term.cursor_x, 1);
        assert_eq!(term.cursor_y, 1);
        assert_eq!(term.screen[0].iter().map(|g|g.c).collect::<String>(), "abc");
        assert_eq!(term.screen[1].iter().map(|g|g.c).collect::<String>(), "d  ");
    }

     #[test]
    fn test_newline() {
        let mut term = create_test_term(5, 3);
        process_str(&mut term, "ab\r\ncd"); // Use \r\n

        assert_eq!(term.cursor_x, 2);
        assert_eq!(term.cursor_y, 1);
        assert_eq!(term.screen[0].iter().map(|g|g.c).collect::<String>(), "ab   ");
        assert_eq!(term.screen[1].iter().map(|g|g.c).collect::<String>(), "cd   ");
    }

    #[test]
    fn test_carriage_return() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abc\rde");

        assert_eq!(term.cursor_x, 2);
        assert_eq!(term.cursor_y, 0);
        assert_eq!(term.screen[0].iter().map(|g|g.c).collect::<String>(), "dec  ");
    }

     #[test]
    fn test_backspace() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abc");
        term.process_byte(0x08); // Backspace

        assert_eq!(term.cursor_x, 2);
        assert_eq!(term.cursor_y, 0);

        process_str(&mut term, "d"); // Type 'd'

        assert_eq!(term.cursor_x, 3);
        assert_eq!(term.screen[0].iter().map(|g|g.c).collect::<String>(), "abd  ");
    }


    #[test]
    fn test_parser_state_csi() {
        let mut term = create_test_term(5, 2);

        term.process_byte(0x1B); // ESC
        assert_eq!(term.parser_state, ParserState::Escape);

        term.process_byte(b'['); // [
        assert_eq!(term.parser_state, ParserState::Csi);

        term.process_byte(b'A'); // A (final byte)
        assert_eq!(term.parser_state, ParserState::Ground);
    }

     #[test]
    fn test_csi_cursor_up() {
        let mut term = create_test_term(5, 3);
        term.cursor_y = 2;
        process_str(&mut term, "\x1b[A");

        assert_eq!(term.cursor_y, 1);
        assert_eq!(term.cursor_x, 0); // Cursor X should not change on simple up/down
    }

     #[test]
    fn test_csi_cursor_up_param() {
        let mut term = create_test_term(5, 5);
        term.cursor_y = 4;
        process_str(&mut term, "\x1b[3A");

        assert_eq!(term.cursor_y, 1);
        assert_eq!(term.cursor_x, 0);
    }

    #[test]
    fn test_csi_cursor_down() {
        let mut term = create_test_term(5, 3);
        term.cursor_y = 0;
        process_str(&mut term, "\x1b[B");

        assert_eq!(term.cursor_y, 1);
        assert_eq!(term.cursor_x, 0);
    }

    #[test]
    fn test_csi_cursor_forward() {
        let mut term = create_test_term(5, 3);
        process_str(&mut term, "\x1b[C"); // Move 1 right

        assert_eq!(term.cursor_y, 0);
        assert_eq!(term.cursor_x, 1);

        process_str(&mut term, "\x1b[2C"); // Move 2 more right

        assert_eq!(term.cursor_x, 3);
    }

    #[test]
    fn test_csi_cursor_backward() {
        let mut term = create_test_term(5, 3);
        term.cursor_x = 4;
        process_str(&mut term, "\x1b[D"); // Move 1 left

        assert_eq!(term.cursor_y, 0);
        assert_eq!(term.cursor_x, 3);

        process_str(&mut term, "\x1b[2D"); // Move 2 more left

        assert_eq!(term.cursor_x, 1);
    }

    #[test]
    fn test_csi_cursor_position() {
        let mut term = create_test_term(10, 5);
        process_str(&mut term, "\x1b[3;4H");

        assert_eq!(term.cursor_y, 2);
        assert_eq!(term.cursor_x, 3);
    }

     #[test]
    fn test_csi_erase_line_to_end() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abcde");
        term.cursor_x = 2;
        process_str(&mut term, "\x1b[K");

        assert_eq!(term.screen[0].iter().map(|g|g.c).collect::<String>(), "ab   ");
        assert_eq!(term.cursor_x, 2);
    }

    #[test]
    fn test_csi_erase_line_to_start() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abcde");
        term.cursor_x = 2;
        process_str(&mut term, "\x1b[1K");

        assert_eq!(term.screen[0].iter().map(|g|g.c).collect::<String>(), "   de");
        assert_eq!(term.cursor_x, 2);
    }

    #[test]
    fn test_csi_erase_whole_line() {
        let mut term = create_test_term(5, 2);
        process_str(&mut term, "abcde");
        term.cursor_x = 2;
        process_str(&mut term, "\x1b[2K");

        assert!(term.screen[0].iter().all(|g| g.c == ' '));
        assert_eq!(term.cursor_x, 2);
    }

    #[test]
    fn test_dec_mode_set_reset() {
        let mut term = create_test_term(10, 5);

        assert!(!term.dec_modes.cursor_keys_app_mode);
        process_str(&mut term, "\x1b[?1h");
        assert!(term.dec_modes.cursor_keys_app_mode);
        process_str(&mut term, "\x1b[?1l");
        assert!(!term.dec_modes.cursor_keys_app_mode);

        assert!(!term.dec_modes.bracketed_paste);
        process_str(&mut term, "\x1b[?2004h");
        assert!(term.dec_modes.bracketed_paste);
        process_str(&mut term, "\x1b[?2004l");
        assert!(!term.dec_modes.bracketed_paste);
    }

    #[test]
    fn test_esc_intermediate_charset() {
        let mut term = create_test_term(10, 5);
        process_str(&mut term, "\x1b(B");

        assert_eq!(term.parser_state, ParserState::Ground);
    }

    #[test]
    fn test_esc_keypad_modes() {
         let mut term = create_test_term(10, 5);
         process_str(&mut term, "\x1b="); // DECKPAM

         assert_eq!(term.parser_state, ParserState::Ground);
         // TODO: Assert keypad state if tracked

         process_str(&mut term, "\x1b>"); // DECKPNM

         assert_eq!(term.parser_state, ParserState::Ground);
         // TODO: Assert keypad state if tracked
    }

    #[test]
    fn test_sgr_reset() {
        let mut term = create_test_term(10, 5);
        term.current_attributes.fg = ColorSpec::Idx(1);
        term.current_attributes.flags = AttrFlags::BOLD;
        process_str(&mut term, "\x1b[m");

        assert_eq!(term.current_attributes, Attributes::default());

        term.current_attributes.fg = ColorSpec::Idx(2);
        process_str(&mut term, "\x1b[0m");

        assert_eq!(term.current_attributes, Attributes::default());
    }

    #[test]
    fn test_sgr_basic_colors() {
        let mut term = create_test_term(10, 5);
        process_str(&mut term, "\x1b[31m"); // Red FG

        assert_eq!(term.current_attributes.fg, ColorSpec::Idx(1));
        assert_eq!(term.current_attributes.bg, ColorSpec::Default);

        process_str(&mut term, "\x1b[42m"); // Green BG

        assert_eq!(term.current_attributes.fg, ColorSpec::Idx(1)); // FG unchanged
        assert_eq!(term.current_attributes.bg, ColorSpec::Idx(2));

        process_str(&mut term, "\x1b[39;49m"); // Default FG/BG

        assert_eq!(term.current_attributes.fg, ColorSpec::Default);
        assert_eq!(term.current_attributes.bg, ColorSpec::Default);
    }

     #[test]
    fn test_sgr_bright_colors() {
        let mut term = create_test_term(10, 5);
        process_str(&mut term, "\x1b[91m"); // Bright Red FG

        assert_eq!(term.current_attributes.fg, ColorSpec::Idx(9));

        process_str(&mut term, "\x1b[102m"); // Bright Green BG

        assert_eq!(term.current_attributes.bg, ColorSpec::Idx(10));
    }

     #[test]
    fn test_sgr_bold_underline() {
        let mut term = create_test_term(10, 5);

        assert!(!term.current_attributes.flags.contains(AttrFlags::BOLD));
        assert!(!term.current_attributes.flags.contains(AttrFlags::UNDERLINE));

        process_str(&mut term, "\x1b[1;4m"); // Bold and Underline

        assert!(term.current_attributes.flags.contains(AttrFlags::BOLD));
        assert!(term.current_attributes.flags.contains(AttrFlags::UNDERLINE));

        process_str(&mut term, "\x1b[22m"); // Normal intensity (removes bold)

        assert!(!term.current_attributes.flags.contains(AttrFlags::BOLD));
        assert!(term.current_attributes.flags.contains(AttrFlags::UNDERLINE));

        process_str(&mut term, "\x1b[24m"); // Not underlined

        assert!(!term.current_attributes.flags.contains(AttrFlags::UNDERLINE));
    }

    #[test]
    fn test_sgr_reverse() {
         let mut term = create_test_term(10, 5);
         term.current_attributes.fg = ColorSpec::Idx(1); // Red
         term.current_attributes.bg = ColorSpec::Idx(4); // Blue

         assert!(!term.current_attributes.flags.contains(AttrFlags::REVERSE));

         process_str(&mut term, "\x1b[7m"); // Reverse

         assert!(term.current_attributes.flags.contains(AttrFlags::REVERSE));
         assert_eq!(term.current_attributes.fg, ColorSpec::Idx(1)); // State doesn't change
         assert_eq!(term.current_attributes.bg, ColorSpec::Idx(4));

         process_str(&mut term, "\x1b[27m"); // Not reversed

         assert!(!term.current_attributes.flags.contains(AttrFlags::REVERSE));
    }

     #[test]
    fn test_sgr_combined() {
        let mut term = create_test_term(10, 5);
        process_str(&mut term, "\x1b[1;31;44m"); // Bold, Red FG, Blue BG

        assert!(term.current_attributes.flags.contains(AttrFlags::BOLD));
        assert!(!term.current_attributes.flags.contains(AttrFlags::UNDERLINE));
        assert_eq!(term.current_attributes.fg, ColorSpec::Idx(1));
        assert_eq!(term.current_attributes.bg, ColorSpec::Idx(4));

        process_str(&mut term, "\x1b[0;4;92m"); // Reset, Underline, Bright Green FG

        assert!(!term.current_attributes.flags.contains(AttrFlags::BOLD));
        assert!(term.current_attributes.flags.contains(AttrFlags::UNDERLINE));
        assert_eq!(term.current_attributes.fg, ColorSpec::Idx(10));
        assert_eq!(term.current_attributes.bg, ColorSpec::Default);
    }

    #[test]
    fn test_sgr_256color() {
        let mut term = create_test_term(10, 5);
        process_str(&mut term, "\x1b[38;5;208m"); // Orange FG

        assert_eq!(term.current_attributes.fg, ColorSpec::Idx(208));

        process_str(&mut term, "\x1b[48;5;18m"); // Dark Blue BG

        assert_eq!(term.current_attributes.bg, ColorSpec::Idx(18));
    }

     #[test]
    fn test_sgr_truecolor() {
        let mut term = create_test_term(10, 5);
        process_str(&mut term, "\x1b[38;2;10;20;30m"); // FG R=10 G=20 B=30

        assert_eq!(term.current_attributes.fg, ColorSpec::Rgb(10, 20, 30));

        process_str(&mut term, "\x1b[48;2;100;150;200m"); // BG R=100 G=150 B=200

        assert_eq!(term.current_attributes.bg, ColorSpec::Rgb(100, 150, 200));
    }
}
