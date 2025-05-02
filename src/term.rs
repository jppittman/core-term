// src/term.rs

use crate::glyph::{Glyph, Attributes, Color, AttrFlags, REPLACEMENT_CHARACTER};
use std::mem;
use std::cmp::min;

// --- Constants ---
/// Default tab stop interval.
const DEFAULT_TAB_INTERVAL: usize = 8;
/// Maximum number of CSI parameters allowed.
const MAX_CSI_PARAMS: usize = 16;
/// Maximum number of CSI intermediate bytes allowed.
const MAX_CSI_INTERMEDIATES: usize = 2;
/// Maximum length of OSC strings (to prevent excessive allocation).
const MAX_OSC_STRING_LEN: usize = 256;


// Helper struct for UTF-8 decoding
struct Utf8Decoder {
    buffer: [u8; 4],
    len: usize,
}

impl Utf8Decoder {
    fn new() -> Self {
        Utf8Decoder { buffer: [0; 4], len: 0 }
    }

    fn decode(&mut self, byte: u8) -> Result<Option<char>, ()> {
        if self.len >= 4 {
            return Err(());
        }
        self.buffer[self.len] = byte;
        self.len += 1;
        match std::str::from_utf8(&self.buffer[0..self.len]) {
            Ok(s) => {
                let expected_len = utf8_char_len(self.buffer[0]);
                if self.len == expected_len {
                    Ok(s.chars().next())
                } else if self.len < expected_len {
                    Ok(None)
                } else {
                    Err(()) // Too many bytes for the leading byte
                }
            }
            Err(e) => {
                if e.error_len().is_none() && self.len < 4 {
                    Ok(None) // Incomplete sequence
                } else {
                    Err(()) // Invalid sequence
                }
            }
        }
    }
}

/// Determines expected UTF-8 character length from the first byte.
fn utf8_char_len(byte: u8) -> usize {
    if byte & 0x80 == 0 { 1 }        // 0xxxxxxx (ASCII)
    else if byte & 0xE0 == 0xC0 { 2 } // 110xxxxx
    else if byte & 0xF0 == 0xE0 { 3 } // 1110xxxx
    else if byte & 0xF8 == 0xF0 { 4 } // 11110xxx
    else { 1 } // Invalid start byte, treat as length 1 for recovery
}

/// Represents the cursor position (0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cursor {
    pub x: usize,
    pub y: usize,
}

/// States for the terminal escape sequence parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParserState {
    Ground, Escape, CSIEntry, CSIParam, CSIIntermediate, CSIIgnore, OSCParse,
}

/// Holds DEC Private Mode settings.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DecModes {
    cursor_keys_app_mode: bool,
}

impl Default for DecModes {
    fn default() -> Self {
        DecModes { cursor_keys_app_mode: false }
    }
}

/// Represents the state of the terminal emulator.
pub struct Term {
    // Made public for direct access from backends (simplifies drawing loops)
    // Consider using getters if encapsulation becomes more important later.
    pub width: usize,
    pub height: usize,
    pub cursor: Cursor,
    // Screen remains private, use get_glyph
    screen: Vec<Vec<Glyph>>,
    alt_screen: Vec<Vec<Glyph>>,
    using_alt_screen: bool,
    saved_cursor: Cursor,
    current_attributes: Attributes,
    default_attributes: Attributes,
    parser_state: ParserState,
    csi_params: Vec<u16>,
    csi_intermediates: Vec<char>,
    utf8_decoder: Utf8Decoder,
    dec_modes: DecModes,
    tabs: Vec<bool>,
}

impl Term {
    /// Creates a new terminal state with given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        let default_attributes = Attributes::default();
        let default_glyph = Glyph { c: ' ', attr: default_attributes };
        let screen = vec![vec![default_glyph; width]; height];
        let alt_screen = vec![vec![default_glyph; width]; height];
        let tabs = (0..width).map(|i| i % DEFAULT_TAB_INTERVAL == 0).collect();

        Term {
            width, height, screen, alt_screen, using_alt_screen: false,
            cursor: Cursor { x: 0, y: 0 },
            saved_cursor: Cursor { x: 0, y: 0 },
            current_attributes: default_attributes,
            default_attributes,
            parser_state: ParserState::Ground,
            csi_params: Vec::with_capacity(MAX_CSI_PARAMS),
            csi_intermediates: Vec::with_capacity(MAX_CSI_INTERMEDIATES),
            utf8_decoder: Utf8Decoder::new(),
            dec_modes: DecModes::default(),
            tabs,
        }
    }

    /// Gets the glyph at the specified coordinates (0-based).
    pub fn get_glyph(&self, x: usize, y: usize) -> Option<&Glyph> {
        let current_screen = if self.using_alt_screen { &self.alt_screen } else { &self.screen };
        current_screen.get(y).and_then(|row| row.get(x))
    }

    // --- Getters for dimensions and cursor (alternative to public fields) ---
    pub fn get_cursor(&self) -> (usize, usize) { (self.cursor.x, self.cursor.y) }
    pub fn get_dimensions(&self) -> (usize, usize) { (self.width, self.height) }

    /// Resizes the terminal emulator state.
    pub fn resize(&mut self, new_width: usize, new_height: usize) {
        if new_width == self.width && new_height == self.height {
            return;
        }

        let default_glyph = Glyph { c: ' ', attr: self.default_attributes };

        let resize_buffer = |buffer: &mut Vec<Vec<Glyph>>, old_height: usize| {
            buffer.resize_with(new_height, || vec![default_glyph; new_width]);
            for y in 0..min(old_height, new_height) {
                buffer[y].resize(new_width, default_glyph);
            }
        };

        let old_height = self.height;
        resize_buffer(&mut self.screen, old_height);
        resize_buffer(&mut self.alt_screen, old_height);

        self.width = new_width;
        self.height = new_height;

        self.cursor.x = min(self.cursor.x, self.width.saturating_sub(1));
        self.cursor.y = min(self.cursor.y, self.height.saturating_sub(1));
        self.saved_cursor.x = min(self.saved_cursor.x, self.width.saturating_sub(1));
        self.saved_cursor.y = min(self.saved_cursor.y, self.height.saturating_sub(1));

        self.tabs = (0..self.width).map(|i| i % DEFAULT_TAB_INTERVAL == 0).collect();
    }

    /// Returns a mutable reference to the currently active screen buffer.
    fn current_screen_mut(&mut self) -> &mut Vec<Vec<Glyph>> {
        if self.using_alt_screen { &mut self.alt_screen } else { &mut self.screen }
    }

    /// Clears CSI parameters and intermediates.
    fn clear_csi_params(&mut self) {
        self.csi_params.clear();
        self.csi_intermediates.clear();
    }

    /// Appends a digit to the last CSI parameter being built.
    fn push_csi_param(&mut self, digit: u16) {
        let current_param = if self.csi_params.is_empty() {
            self.csi_params.push(0);
            self.csi_params.last_mut().unwrap()
        } else {
            self.csi_params.last_mut().unwrap()
        };
        *current_param = current_param.saturating_mul(10).saturating_add(digit);
    }

    /// Moves to the next CSI parameter slot (appends a 0).
    fn next_csi_param(&mut self) {
         if self.csi_params.len() < MAX_CSI_PARAMS {
             self.csi_params.push(0);
         }
    }

     /// Gets the nth CSI parameter (0-based index), defaulting if absent/0.
    fn get_csi_param(&self, index: usize, default: u16) -> u16 {
        match self.csi_params.get(index) {
            Some(&p) if p > 0 => p,
            _ => default,
        }
    }

     /// Gets the nth CSI parameter (0-based index), returning 0 if absent.
     fn get_csi_param_or_0(&self, index: usize) -> u16 {
        *self.csi_params.get(index).unwrap_or(&0)
    }

    // --- Scrolling ---
    /// Scrolls the active screen content up by `lines`.
    fn scroll_up(&mut self, lines: usize) {
        // Borrow immutable fields first
        let height = self.height;
        let default_glyph = Glyph { c: ' ', attr: self.default_attributes }; // Copy default attributes

        // Calculate scroll amount
        let lines_to_scroll = min(lines, height);
        if lines_to_scroll == 0 || height == 0 {
            return;
        }

        // Now borrow screen mutably
        let screen = self.current_screen_mut();
        screen.rotate_left(lines_to_scroll);
        for y in (height - lines_to_scroll)..height {
            screen[y].fill(default_glyph);
        }
    }

    // --- Cursor Movement ---
    /// Moves the cursor relative to its current position, clamping to bounds.
    fn move_cursor(&mut self, dx: isize, dy: isize) {
        // Read dimensions before modifying cursor
        let width = self.width;
        let height = self.height;
        let new_x = (self.cursor.x as isize + dx).max(0) as usize;
        let new_y = (self.cursor.y as isize + dy).max(0) as usize;
        self.cursor.x = min(new_x, width.saturating_sub(1));
        self.cursor.y = min(new_y, height.saturating_sub(1));
    }

    /// Sets the cursor to an absolute position (0-based), clamping to bounds.
    fn set_cursor_pos(&mut self, x: usize, y: usize) {
        // Read dimensions before modifying cursor
        let width = self.width;
        let height = self.height;
        self.cursor.x = min(x, width.saturating_sub(1));
        self.cursor.y = min(y, height.saturating_sub(1));
    }

    /// Handles Line Feed (LF).
    fn newline(&mut self) {
        let height = self.height; // Read height before potentially calling scroll_up (which needs &mut self)
        let next_y = self.cursor.y + 1;
        if next_y >= height {
            self.scroll_up(1);
        } else {
            self.cursor.y = next_y;
        }
        self.cursor.x = 0;
    }

    /// Handles Carriage Return (CR).
    fn carriage_return(&mut self) {
        self.cursor.x = 0;
    }

    /// Handles Backspace (BS).
    fn backspace(&mut self) {
        if self.cursor.x > 0 {
            self.cursor.x -= 1;
        }
    }

    /// Handles Horizontal Tab (HT).
    fn tab(&mut self) {
        // Read immutable fields first
        let current_x = self.cursor.x;
        let width = self.width;
        let tabs = &self.tabs; // Borrow immutably

        let next_tab_pos = tabs.iter()
            .skip(current_x + 1)
            .position(|&is_stop| is_stop);

        match next_tab_pos {
            Some(relative_pos) => {
                let absolute_pos = current_x + 1 + relative_pos;
                self.cursor.x = min(absolute_pos, width.saturating_sub(1));
            }
            None => {
                self.cursor.x = width.saturating_sub(1);
            }
        }
    }


    // --- Erasing Functions ---
    /// Fills cells with spaces using current attributes.
    fn fill_range(&mut self, y: usize, x_start: usize, x_end: usize) {
        // Read immutable fields first
        let height = self.height;
        let width = self.width;
        let fill_glyph = Glyph { c: ' ', attr: self.current_attributes }; // Copy current attributes

        // Check bounds before mutable borrow
        if y >= height || x_start >= x_end || x_start >= width {
            return;
        }

        // Now borrow screen mutably
        let screen = self.current_screen_mut();
        let clamped_end = min(x_end, width);
        for x in x_start..clamped_end {
            screen[y][x] = fill_glyph;
        }
    }

    /// Erases from cursor to end of line (EL 0).
    fn erase_line_to_end(&mut self) {
        // Read immutable fields needed by fill_range
        let y = self.cursor.y;
        let x = self.cursor.x;
        let width = self.width;
        self.fill_range(y, x, width);
    }

    /// Erases from start of line to cursor (inclusive) (EL 1).
    fn erase_line_to_start(&mut self) {
        // Read immutable fields needed by fill_range
        let y = self.cursor.y;
        let x = self.cursor.x;
        self.fill_range(y, 0, x + 1);
    }

    /// Erases the entire current line (EL 2).
    fn erase_whole_line(&mut self) {
        // Read immutable fields needed by fill_range
        let y = self.cursor.y;
        let width = self.width;
        self.fill_range(y, 0, width);
    }

    /// Erases from cursor to end of display (ED 0).
    fn erase_display_to_end(&mut self) {
        // Read immutable fields first
        let cursor_y = self.cursor.y;
        let height = self.height;
        let fill_glyph = Glyph { c: ' ', attr: self.current_attributes }; // Copy attributes

        // Erase rest of current line
        self.erase_line_to_end();

        // Now borrow screen mutably for lines below
        let screen = self.current_screen_mut();
        for y in (cursor_y + 1)..height {
            screen[y].fill(fill_glyph);
        }
    }

    /// Erases from start of display to cursor (inclusive) (ED 1).
    fn erase_display_to_start(&mut self) {
        // Read immutable fields first
        let cursor_y = self.cursor.y;
        let height = self.height; // Only needed for loop bound below
        let fill_glyph = Glyph { c: ' ', attr: self.current_attributes }; // Copy attributes

        // Erase beginning of current line
        self.erase_line_to_start();

        // Now borrow screen mutably for lines above
        let screen = self.current_screen_mut();
        for y in 0..cursor_y { // Iterate up to (but not including) cursor line
             if y < height { // Ensure y is within bounds
                 screen[y].fill(fill_glyph);
             }
        }
    }

    /// Erases the entire display (ED 2).
    fn erase_whole_display(&mut self) {
        // Read immutable fields first
        let height = self.height;
        let fill_glyph = Glyph { c: ' ', attr: self.current_attributes }; // Copy attributes

        // Now borrow screen mutably
        let screen = self.current_screen_mut();
        for y in 0..height {
            screen[y].fill(fill_glyph);
        }
        // self.set_cursor_pos(0, 0); // Optionally move cursor home
    }


    // --- Character Handling ---
    /// Handles a printable character decoded from the input stream.
    fn handle_printable(&mut self, c: char) {
        let char_width = 1; // TODO: Implement proper Unicode width
        let width = self.width; // Read width before potential mutable borrow in newline
        let height = self.height; // Read height before potential mutable borrow
        let current_attributes = self.current_attributes; // Copy attributes

        // Check wrap condition using local width
        if self.cursor.x + char_width > width {
             if self.cursor.x < width {
                 self.newline(); // Needs &mut self
             } else {
                 self.newline(); // Needs &mut self
             }
        }

        // Check bounds again using local height/width *after* potential newline
        if self.cursor.y < height && self.cursor.x < width {
            let y = self.cursor.y;
            let x = self.cursor.x;
            // Borrow screen mutably *only* for the write
            let screen = self.current_screen_mut();
            screen[y][x] = Glyph { c, attr: current_attributes }; // Use copied attributes

            if char_width > 0 {
                self.cursor.x += char_width;
            }
        }
    }


    // --- Escape Sequence Parsing States ---

    /// Processes a single byte received from the PTY or input source.
    pub fn process_byte(&mut self, byte: u8) {
        match self.parser_state {
            ParserState::Ground => self.handle_ground_byte(byte),
            ParserState::Escape => self.handle_escape_byte(byte),
            ParserState::CSIEntry => self.handle_csi_entry_byte(byte),
            ParserState::CSIParam => self.handle_csi_param_byte(byte),
            ParserState::CSIIntermediate => self.handle_csi_intermediate_byte(byte),
            ParserState::CSIIgnore => self.handle_csi_ignore_byte(byte),
            ParserState::OSCParse => self.handle_osc_parse_byte(byte),
        }
    }

    /// Handles bytes in the Ground state (normal operation).
    fn handle_ground_byte(&mut self, byte: u8) {
        match byte {
            0x08 => self.backspace(),
            0x09 => self.tab(),
            0x0A | 0x0B | 0x0C => self.newline(),
            0x0D => self.carriage_return(),
            0x00..=0x07 | 0x0E..=0x1A | 0x1C..=0x1F => { /* Ignore other C0 */ }
            0x1B => { self.parser_state = ParserState::Escape; self.clear_csi_params(); }
            _ => {
                match self.utf8_decoder.decode(byte) {
                    Ok(Some(c)) => { self.handle_printable(c); self.utf8_decoder = Utf8Decoder::new(); }
                    Ok(None) => { /* Need more bytes */ }
                    Err(_) => { self.handle_printable(REPLACEMENT_CHARACTER); self.utf8_decoder = Utf8Decoder::new(); }
                }
            }
        }
    }

    /// Handles bytes after an ESC (0x1B) has been received.
    fn handle_escape_byte(&mut self, byte: u8) {
        match byte {
            b'[' => self.parser_state = ParserState::CSIEntry,
            b']' => self.parser_state = ParserState::OSCParse,
            b'c' => { // RIS
                // Read default attributes before mutable borrow
                let default_glyph = Glyph { c: ' ', attr: self.default_attributes };
                let screen = self.current_screen_mut();
                for row in screen.iter_mut() { row.fill(default_glyph); }
                self.cursor = Cursor { x: 0, y: 0 };
                self.current_attributes = self.default_attributes;
                self.dec_modes = DecModes::default();
                self.parser_state = ParserState::Ground;
            }
            b'7' => { self.saved_cursor = self.cursor; self.parser_state = ParserState::Ground; }
            b'8' => { self.cursor = self.saved_cursor; self.parser_state = ParserState::Ground; }
            b'(' | b')' | b'*' | b'+' => self.parser_state = ParserState::Ground, // Ignore SCS
            b'=' | b'>' => self.parser_state = ParserState::Ground, // Ignore keypad modes
            _ => self.parser_state = ParserState::Ground,
        }
    }

    /// Handles the first byte after ESC [.
    fn handle_csi_entry_byte(&mut self, byte: u8) {
        match byte {
            // Cast to u16 for push_csi_param
            b'0'..=b'9' => { self.push_csi_param((byte - b'0') as u16); self.parser_state = ParserState::CSIParam; }
            b';' => { self.next_csi_param(); self.parser_state = ParserState::CSIParam; }
            b'?' | b'>' | b'!' | b':' => {
                if self.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                     self.csi_intermediates.push(byte as char);
                     self.parser_state = ParserState::CSIIntermediate;
                 } else { self.parser_state = ParserState::CSIIgnore; }
            }
            0x40..=0x7E => { self.csi_dispatch(byte); self.parser_state = ParserState::Ground; }
            _ => self.parser_state = ParserState::Ground,
        }
    }

    /// Handles CSI parameter bytes (digits, ';').
    fn handle_csi_param_byte(&mut self, byte: u8) {
        match byte {
            // Cast to u16 for push_csi_param
            b'0'..=b'9' => self.push_csi_param((byte - b'0') as u16),
            b';' => self.next_csi_param(),
            0x20..=0x2F => {
                 if self.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                     self.csi_intermediates.push(byte as char);
                     self.parser_state = ParserState::CSIIntermediate;
                 } else { self.parser_state = ParserState::CSIIgnore; }
            }
            0x40..=0x7E => { self.csi_dispatch(byte); self.parser_state = ParserState::Ground; }
            _ => self.parser_state = ParserState::Ground,
        }
    }

    /// Handles CSI intermediate bytes (after params, before final byte).
    fn handle_csi_intermediate_byte(&mut self, byte: u8) {
        match byte {
            0x20..=0x2F => {
                if self.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                    self.csi_intermediates.push(byte as char);
                } else { self.parser_state = ParserState::CSIIgnore; }
            }
            0x40..=0x7E => { self.csi_dispatch(byte); self.parser_state = ParserState::Ground; }
            _ => self.parser_state = ParserState::Ground,
        }
    }

    /// Handles bytes when ignoring the rest of a CSI sequence.
    fn handle_csi_ignore_byte(&mut self, byte: u8) {
        if (0x40..=0x7E).contains(&byte) {
            self.parser_state = ParserState::Ground;
        }
    }

    /// Handles bytes during OSC sequence parsing (ESC ] ... ST/BEL).
    fn handle_osc_parse_byte(&mut self, byte: u8) {
        // TODO: Implement proper OSC string collection and handling
        match byte {
            0x07 | 0x1B => self.parser_state = ParserState::Ground, // BEL or ESC terminates
            0x00..=0x1A | 0x1C..=0x1F => {} // Ignore C0
            _ => { /* Collect byte if implementing OSC */ }
        }
    }


    /// Dispatches action based on the final byte of a CSI sequence.
    fn csi_dispatch(&mut self, final_byte: u8) {
        let is_private = self.csi_intermediates.first() == Some(&'?');

        match final_byte {
            b'A' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(0, -n); }
            b'B' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(0, n); }
            b'C' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(n, 0); }
            b'D' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(-n, 0); }
            b'E' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(0, n); self.cursor.x = 0; }
            b'F' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(0, -n); self.cursor.x = 0; }
            b'G' => { let n = self.get_csi_param(0, 1); self.cursor.x = min(n.saturating_sub(1) as usize, self.width.saturating_sub(1)); }
            b'H' | b'f' => { let y = self.get_csi_param(0, 1); let x = self.get_csi_param(1, 1); self.set_cursor_pos(x.saturating_sub(1) as usize, y.saturating_sub(1) as usize); }
            b'J' => { match self.get_csi_param_or_0(0) { 0 => self.erase_display_to_end(), 1 => self.erase_display_to_start(), 2 => self.erase_whole_display(), _ => {} } }
            b'K' => { match self.get_csi_param_or_0(0) { 0 => self.erase_line_to_end(), 1 => self.erase_line_to_start(), 2 => self.erase_whole_line(), _ => {} } }
            b'm' => self.handle_sgr(),
            b's' => self.saved_cursor = self.cursor,
            b'u' => self.cursor = self.saved_cursor,
            b'h' if is_private => self.handle_dec_mode_set(true),
            b'l' if is_private => self.handle_dec_mode_set(false),
            b'h' if is_private && self.get_csi_param_or_0(0) == 1049 => {
                 if !self.using_alt_screen {
                     mem::swap(&mut self.screen, &mut self.alt_screen);
                     self.using_alt_screen = true;
                     self.erase_whole_display();
                     self.set_cursor_pos(0, 0);
                 }
             }
             b'l' if is_private && self.get_csi_param_or_0(0) == 1049 => {
                 if self.using_alt_screen {
                     mem::swap(&mut self.screen, &mut self.alt_screen);
                     self.using_alt_screen = false;
                 }
             }
            _ => { /* Ignore unknown */ }
        }
    }

    /// Handles SGR (Select Graphic Rendition) codes (CSI ... m).
    fn handle_sgr(&mut self) {
        if self.csi_params.is_empty() || self.get_csi_param_or_0(0) == 0 {
            self.current_attributes = self.default_attributes;
            if self.csi_params.len() > 1 { self.process_sgr_codes(1); }
            return;
        }
        self.process_sgr_codes(0);
    }

    /// Processes SGR codes starting from a given index in `csi_params`.
    fn process_sgr_codes(&mut self, start_index: usize) {
        let mut i = start_index;
        while i < self.csi_params.len() {
            let code = self.get_csi_param_or_0(i);
            match code {
                0 => self.current_attributes = self.default_attributes,
                1 => self.current_attributes.flags |= AttrFlags::BOLD,
                3 => self.current_attributes.flags |= AttrFlags::ITALIC,
                4 => self.current_attributes.flags |= AttrFlags::UNDERLINE,
                7 => self.current_attributes.flags |= AttrFlags::REVERSE,
                8 => self.current_attributes.flags |= AttrFlags::HIDDEN,
                9 => self.current_attributes.flags |= AttrFlags::STRIKETHROUGH,
                22 => self.current_attributes.flags &= !AttrFlags::BOLD,
                23 => self.current_attributes.flags &= !AttrFlags::ITALIC,
                24 => self.current_attributes.flags &= !AttrFlags::UNDERLINE,
                27 => self.current_attributes.flags &= !AttrFlags::REVERSE,
                28 => self.current_attributes.flags &= !AttrFlags::HIDDEN,
                29 => self.current_attributes.flags &= !AttrFlags::STRIKETHROUGH,
                30..=37 => self.current_attributes.fg = Color::Idx((code - 30) as u8),
                38 => {
                    i += 1;
                    match self.get_csi_param_or_0(i) {
                        5 => { i += 1; let idx = self.get_csi_param_or_0(i) as u8; self.current_attributes.fg = Color::Idx(idx); }
                        2 => {
                            if i + 3 < self.csi_params.len() {
                                let r = self.get_csi_param_or_0(i + 1) as u8; let g = self.get_csi_param_or_0(i + 2) as u8; let b = self.get_csi_param_or_0(i + 3) as u8;
                                self.current_attributes.fg = Color::Rgb(r, g, b); i += 3;
                            } else { i = self.csi_params.len(); }
                        }
                        _ => { i += 1; } // Skip potential value
                    }
                }
                39 => self.current_attributes.fg = self.default_attributes.fg,
                40..=47 => self.current_attributes.bg = Color::Idx((code - 40) as u8),
                48 => {
                    i += 1;
                    match self.get_csi_param_or_0(i) {
                        5 => { i += 1; let idx = self.get_csi_param_or_0(i) as u8; self.current_attributes.bg = Color::Idx(idx); }
                        2 => {
                            if i + 3 < self.csi_params.len() {
                                let r = self.get_csi_param_or_0(i + 1) as u8; let g = self.get_csi_param_or_0(i + 2) as u8; let b = self.get_csi_param_or_0(i + 3) as u8;
                                self.current_attributes.bg = Color::Rgb(r, g, b); i += 3;
                            } else { i = self.csi_params.len(); }
                        }
                         _ => { i += 1; } // Skip potential value
                    }
                }
                49 => self.current_attributes.bg = self.default_attributes.bg,
                90..=97 => self.current_attributes.fg = Color::Idx((code - 90 + 8) as u8),
                100..=107 => self.current_attributes.bg = Color::Idx((code - 100 + 8) as u8),
                _ => { /* Ignore */ }
            }
            i += 1;
        }
    }

    /// Handles DEC Private Mode Set/Reset sequences (CSI ? Pn h/l).
    fn handle_dec_mode_set(&mut self, enable: bool) {
        for i in 0..self.csi_params.len() {
             let mode = self.get_csi_param_or_0(i);
             match mode {
                1 => self.dec_modes.cursor_keys_app_mode = enable,
                1049 => { /* Handled in csi_dispatch */ }
                _ => { /* Ignore unknown */ }
            }
        }
    }

    /// Processes a slice of bytes received from the PTY or input source.
    pub fn process_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.process_byte(byte);
        }
    }

    /// Helper to get screen content as string for a specific line (for testing).
    #[cfg(test)]
    fn screen_to_string_line(&self, y: usize) -> String {
        let current_screen = if self.using_alt_screen { &self.alt_screen } else { &self.screen };
        current_screen.get(y)
            .map(|row| row.iter().map(|g| g.c).collect())
            .unwrap_or_default()
    }
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*; // Import items from outer module

    /// Helper to create a Term instance and process initial bytes.
    fn term_with_bytes(w: usize, h: usize, bytes: &[u8]) -> Term {
        let mut term = Term::new(w, h);
        term.process_bytes(bytes);
        term
    }

    // --- Basic Functionality Tests ---
    #[test]
    fn test_printable_chars() {
        let term = term_with_bytes(5, 1, b"abc");
        assert_eq!(term.cursor.x, 3);
        assert_eq!(term.cursor.y, 0);
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'a');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, 'b');
        assert_eq!(term.get_glyph(2, 0).unwrap().c, 'c');
        assert_eq!(term.get_glyph(3, 0).unwrap().c, ' ');
    }

    #[test]
    fn test_line_wrap() {
        let term = term_with_bytes(3, 2, b"abcde");
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 1);
        assert_eq!(term.screen_to_string_line(0), "abc");
        assert_eq!(term.screen_to_string_line(1), "de ");
    }

    #[test]
    fn test_newline() {
        let term = term_with_bytes(5, 2, b"ab\ncd");
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 1);
        assert_eq!(term.screen_to_string_line(0), "ab   ");
        assert_eq!(term.screen_to_string_line(1), "cd   ");
    }

    #[test]
    fn test_carriage_return() {
        let term = term_with_bytes(5, 1, b"abc\rde");
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 0);
        assert_eq!(term.screen_to_string_line(0), "dec  ");
    }

    #[test]
    fn test_backspace() {
        let term = term_with_bytes(5, 1, b"abc\x08d");
        assert_eq!(term.cursor.x, 3);
        assert_eq!(term.screen_to_string_line(0), "abd  ");
    }

    // --- Parser State Tests ---
    #[test]
    fn test_parser_state_csi() {
        let mut term = Term::new(10, 5);
        term.process_byte(0x1b);
        assert_eq!(term.parser_state, ParserState::Escape);
        term.process_byte(b'[');
        assert_eq!(term.parser_state, ParserState::CSIEntry);
        term.process_byte(b'1');
        assert_eq!(term.parser_state, ParserState::CSIParam);
        term.process_byte(b';');
        assert_eq!(term.parser_state, ParserState::CSIParam);
        term.process_byte(b'2');
        assert_eq!(term.parser_state, ParserState::CSIParam);
        term.process_byte(b'H');
        assert_eq!(term.parser_state, ParserState::Ground);
        assert_eq!(term.cursor.x, 1);
        assert_eq!(term.cursor.y, 0);
    }

    #[test]
    fn test_esc_keypad_modes() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b=");
        assert_eq!(term.parser_state, ParserState::Ground);
        term.process_bytes(b"\x1b>");
        assert_eq!(term.parser_state, ParserState::Ground);
    }

    #[test]
    fn test_esc_intermediate_charset() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b(B");
        assert_eq!(term.parser_state, ParserState::Ground);
    }

    // --- CSI Cursor Movement Tests ---
    #[test]
    fn test_csi_cursor_up() {
        let mut term = Term::new(10, 5);
        term.set_cursor_pos(3, 3);
        term.process_bytes(b"\x1b[A");
        assert_eq!(term.cursor, Cursor { x: 3, y: 2 });
    }

    #[test]
    fn test_csi_cursor_up_param() {
        let mut term = Term::new(10, 5);
        term.set_cursor_pos(3, 3);
        term.process_bytes(b"\x1b[2A");
        assert_eq!(term.cursor, Cursor { x: 3, y: 1 });
    }

    #[test]
    fn test_csi_cursor_down() {
        let mut term = Term::new(10, 5);
        term.set_cursor_pos(3, 1);
        term.process_bytes(b"\x1b[B");
        assert_eq!(term.cursor, Cursor { x: 3, y: 2 });
        term.process_bytes(b"\x1b[0B");
        assert_eq!(term.cursor, Cursor { x: 3, y: 3 });
        term.process_bytes(b"\x1b[1B");
        assert_eq!(term.cursor, Cursor { x: 3, y: 4 });
    }

    #[test]
    fn test_csi_cursor_forward() {
        let mut term = Term::new(10, 5);
        term.set_cursor_pos(1, 1);
        term.process_bytes(b"\x1b[C");
        assert_eq!(term.cursor, Cursor { x: 2, y: 1 });
        term.process_bytes(b"\x1b[2C");
        assert_eq!(term.cursor, Cursor { x: 4, y: 1 });
    }

    #[test]
    fn test_csi_cursor_backward() {
        let mut term = Term::new(10, 5);
        term.set_cursor_pos(3, 1);
        term.process_bytes(b"\x1b[D");
        assert_eq!(term.cursor, Cursor { x: 2, y: 1 });
        term.process_bytes(b"\x1b[2D");
        assert_eq!(term.cursor, Cursor { x: 0, y: 1 });
        term.process_bytes(b"\x1b[D");
        assert_eq!(term.cursor, Cursor { x: 0, y: 1 });
    }

    #[test]
    fn test_csi_cursor_position() {
        let term = term_with_bytes(10, 5, b"\x1b[3;4H");
        assert_eq!(term.cursor, Cursor { x: 3, y: 2 });
        let term2 = term_with_bytes(10, 5, b"\x1b[H");
        assert_eq!(term2.cursor, Cursor { x: 0, y: 0 });
        let term3 = term_with_bytes(10, 5, b"\x1b[;5H");
        assert_eq!(term3.cursor, Cursor { x: 4, y: 0 });
        let term4 = term_with_bytes(10, 5, b"\x1b[5;H");
        assert_eq!(term4.cursor, Cursor { x: 0, y: 4 });
    }

    // --- CSI Erase Tests ---
    #[test]
    fn test_csi_erase_line_to_end() {
        let mut term = Term::new(5, 1);
        term.process_bytes(b"abcde");
        term.set_cursor_pos(2, 0);
        term.process_bytes(b"\x1b[K");
        assert_eq!(term.screen_to_string_line(0), "ab   ");
        assert_eq!(term.cursor.x, 2);
    }

    #[test]
    fn test_csi_erase_line_to_start() {
        let mut term = Term::new(5, 1);
        term.process_bytes(b"abcde");
        term.set_cursor_pos(2, 0);
        term.process_bytes(b"\x1b[1K");
        assert_eq!(term.screen_to_string_line(0), "   de");
        assert_eq!(term.cursor.x, 2);
    }

    #[test]
    fn test_csi_erase_whole_line() {
        let mut term = Term::new(5, 1);
        term.process_bytes(b"abcde");
        term.set_cursor_pos(2, 0);
        term.process_bytes(b"\x1b[2K");
        assert_eq!(term.screen_to_string_line(0), "     ");
        assert_eq!(term.cursor.x, 2);
    }

    // --- DEC Private Mode Tests ---
    #[test]
    fn test_dec_mode_set_reset() {
        let mut term = Term::new(10, 5);
        assert!(!term.dec_modes.cursor_keys_app_mode);
        term.process_bytes(b"\x1b[?1h");
        assert!(term.dec_modes.cursor_keys_app_mode);
        term.process_bytes(b"\x1b[?1l");
        assert!(!term.dec_modes.cursor_keys_app_mode);
    }

    // --- SGR Tests ---
    #[test]
    fn test_sgr_reset() {
        let mut term = Term::new(10, 5);
        term.current_attributes.fg = Color::Idx(1);
        term.current_attributes.flags |= AttrFlags::BOLD;
        term.process_bytes(b"\x1b[m");
        assert_eq!(term.current_attributes, term.default_attributes);
        term.current_attributes.fg = Color::Idx(1);
        term.process_bytes(b"\x1b[0m");
        assert_eq!(term.current_attributes, term.default_attributes);
    }

    #[test]
    fn test_sgr_basic_colors() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[31m");
        assert_eq!(term.current_attributes.fg, Color::Idx(1));
        term.process_bytes(b"\x1b[42m");
        assert_eq!(term.current_attributes.bg, Color::Idx(2));
        term.process_bytes(b"\x1b[39m");
        assert_eq!(term.current_attributes.fg, Color::Default);
        term.process_bytes(b"\x1b[49m");
        assert_eq!(term.current_attributes.bg, Color::Default);
    }

    #[test]
    fn test_sgr_bright_colors() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[91m");
        assert_eq!(term.current_attributes.fg, Color::Idx(9));
        term.process_bytes(b"\x1b[102m");
        assert_eq!(term.current_attributes.bg, Color::Idx(10));
    }

    #[test]
    fn test_sgr_bold_underline() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[1m");
        assert!(term.current_attributes.flags.contains(AttrFlags::BOLD));
        term.process_bytes(b"\x1b[4m");
        assert!(term.current_attributes.flags.contains(AttrFlags::UNDERLINE));
        term.process_bytes(b"\x1b[22m");
        assert!(!term.current_attributes.flags.contains(AttrFlags::BOLD));
        assert!(term.current_attributes.flags.contains(AttrFlags::UNDERLINE));
        term.process_bytes(b"\x1b[24m");
        assert!(!term.current_attributes.flags.contains(AttrFlags::UNDERLINE));
    }

    #[test]
    fn test_sgr_reverse() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[7m");
        assert!(term.current_attributes.flags.contains(AttrFlags::REVERSE));
        term.process_bytes(b"\x1b[27m");
        assert!(!term.current_attributes.flags.contains(AttrFlags::REVERSE));
    }

    #[test]
    fn test_sgr_combined() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[1;31;44m");
        assert!(term.current_attributes.flags.contains(AttrFlags::BOLD));
        assert_eq!(term.current_attributes.fg, Color::Idx(1));
        assert_eq!(term.current_attributes.bg, Color::Idx(4));
        term.process_bytes(b"\x1b[0m");
        assert_eq!(term.current_attributes, term.default_attributes);
    }

    #[test]
    fn test_sgr_256color() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[38;5;208m");
        assert_eq!(term.current_attributes.fg, Color::Idx(208));
        term.process_bytes(b"\x1b[48;5;75m");
        assert_eq!(term.current_attributes.bg, Color::Idx(75));
    }

    #[test]
    fn test_sgr_truecolor() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[38;2;10;20;30m");
        assert_eq!(term.current_attributes.fg, Color::Rgb(10, 20, 30));
        term.process_bytes(b"\x1b[48;2;100;150;200m");
        assert_eq!(term.current_attributes.bg, Color::Rgb(100, 150, 200));
    }

    // --- UTF-8 Tests ---
    #[test]
    fn test_utf8_basic() {
        let term = term_with_bytes(10, 1, "你好".as_bytes());
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.get_glyph(0, 0).unwrap().c, '你');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, '好');
    }

    #[test]
    fn test_utf8_mixed() {
        let term = term_with_bytes(10, 1, "a你b好c".as_bytes());
        assert_eq!(term.cursor.x, 5);
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'a');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, '你');
        assert_eq!(term.get_glyph(2, 0).unwrap().c, 'b');
        assert_eq!(term.get_glyph(3, 0).unwrap().c, '好');
        assert_eq!(term.get_glyph(4, 0).unwrap().c, 'c');
    }

    #[test]
    fn test_utf8_invalid_sequence() {
        let term = term_with_bytes(10, 1, &[0x61, 0x80, 0x62]);
        assert_eq!(term.cursor.x, 3);
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'a');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, REPLACEMENT_CHARACTER);
        assert_eq!(term.get_glyph(2, 0).unwrap().c, 'b');
    }

    #[test]
    fn test_utf8_incomplete_sequence_at_end() {
        let term = term_with_bytes(10, 1, &[0x61, 0xE4]);
        assert_eq!(term.cursor.x, 1);
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'a');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, ' ');
        assert_eq!(term.utf8_decoder.len, 1);
        assert_eq!(term.utf8_decoder.buffer[0], 0xE4);
    }

    #[test]
    fn test_utf8_incomplete_sequence_then_continue() {
        let mut term = Term::new(10, 1);
        term.process_byte(0x61);
        term.process_byte(0xE4);
        assert_eq!(term.cursor.x, 1);
        assert_eq!(term.utf8_decoder.len, 1);
        term.process_byte(0xBD);
        assert_eq!(term.cursor.x, 1);
        assert_eq!(term.utf8_decoder.len, 2);
        term.process_byte(0xA0);
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.get_glyph(1, 0).unwrap().c, '你');
        assert_eq!(term.utf8_decoder.len, 0);
    }

    #[test]
    fn test_utf8_with_csi() {
        let term = term_with_bytes(10, 1, b"a\x1b[C\xe4\xbd\xa0\xe5\xa5\xbd");
        assert_eq!(term.cursor.x, 4);
        assert_eq!(term.screen_to_string_line(0), "a 你好      ");
    }

    // --- Resize Test ---
    #[test]
    fn test_resize() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"123\n456");
        term.set_cursor_pos(2, 1);

        // Shrink
        term.resize(5, 3);
        assert_eq!(term.width, 5);
        assert_eq!(term.height, 3);
        assert_eq!(term.screen.len(), 3);
        assert_eq!(term.screen[0].len(), 5);
        assert_eq!(term.screen[1].len(), 5);
        assert_eq!(term.screen[2].len(), 5);
        assert_eq!(term.screen_to_string_line(0), "123  ");
        assert_eq!(term.screen_to_string_line(1), "456  ");
        assert_eq!(term.screen_to_string_line(2), "     ");
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 1);

        // Grow
        term.resize(15, 7);
        assert_eq!(term.width, 15);
        assert_eq!(term.height, 7);
        assert_eq!(term.screen.len(), 7);
        assert_eq!(term.screen[0].len(), 15);
        assert_eq!(term.screen_to_string_line(0), "123            ");
        assert_eq!(term.screen_to_string_line(1), "456            ");
        assert_eq!(term.screen_to_string_line(2), "               ");
        assert_eq!(term.screen_to_string_line(3), "               ");
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 1);
    }

}
