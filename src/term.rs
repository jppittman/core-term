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
            // Prevent buffer overflow if more than 4 bytes are pushed
            // without forming a valid character.
            self.len = 0; // Reset decoder
            return Err(());
        }
        self.buffer[self.len] = byte;
        self.len += 1;

        match std::str::from_utf8(&self.buffer[0..self.len]) {
            Ok(s) => {
                // Successfully decoded a character.
                // Check if it's complete based on the first byte.
                let expected_len = utf8_char_len(self.buffer[0]);
                if self.len == expected_len {
                    self.len = 0; // Reset for next character
                    Ok(s.chars().next()) // Should always contain one char
                } else if self.len < expected_len {
                    // Need more bytes for this character.
                    Ok(None)
                } else {
                    // This case should theoretically not be reached if expected_len
                    // is calculated correctly, but handle defensively.
                    self.len = 0; // Reset decoder
                    Err(()) // Too many bytes for the leading byte
                }
            }
            Err(e) => {
                // Check if the error is due to an incomplete sequence
                // and if we still have buffer space.
                if e.error_len().is_none() && self.len < 4 {
                    Ok(None) // Incomplete sequence, wait for more bytes.
                } else {
                    // Invalid sequence (e.g., overlong encoding, unexpected continuation byte)
                    // or buffer full with incomplete sequence.
                    self.len = 0; // Reset decoder
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
    // Add other DEC modes here as needed (e.g., mouse modes, alt screen)
}

impl Default for DecModes {
    fn default() -> Self {
        DecModes {
            cursor_keys_app_mode: false,
        }
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

        // Helper closure to resize a screen buffer
        let resize_buffer = |buffer: &mut Vec<Vec<Glyph>>, old_height: usize| {
            // Adjust number of rows
            buffer.resize_with(new_height, || vec![default_glyph; new_width]);
            // Adjust width of existing rows
            for y in 0..min(old_height, new_height) {
                buffer[y].resize(new_width, default_glyph);
            }
        };

        let old_height = self.height;
        resize_buffer(&mut self.screen, old_height);
        resize_buffer(&mut self.alt_screen, old_height);

        self.width = new_width;
        self.height = new_height;

        // Clamp cursor and saved cursor positions to new bounds
        self.cursor.x = min(self.cursor.x, self.width.saturating_sub(1));
        self.cursor.y = min(self.cursor.y, self.height.saturating_sub(1));
        self.saved_cursor.x = min(self.saved_cursor.x, self.width.saturating_sub(1));
        self.saved_cursor.y = min(self.saved_cursor.y, self.height.saturating_sub(1));

        // Recalculate tab stops
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
        // Ensure we don't exceed the maximum number of parameters
        if self.csi_params.len() > MAX_CSI_PARAMS {
             return;
        }
        // If no parameters exist yet, add the first one (initialized to 0).
        let current_param = if self.csi_params.is_empty() {
            self.csi_params.push(0);
            self.csi_params.last_mut().unwrap() // Safe unwrap after push
        } else {
            // Get the last parameter, assuming it exists.
            // If csi_params was empty, the above arm would handle it.
            self.csi_params.last_mut().unwrap()
        };
        // Add the digit, preventing overflow with saturating arithmetic.
        *current_param = current_param.saturating_mul(10).saturating_add(digit);
    }


    /// Moves to the next CSI parameter slot (appends a 0).
    fn next_csi_param(&mut self) {
         if self.csi_params.is_empty() {
             // If the first thing is a ';', it means the first param is default/empty.
             // Add a placeholder for the first param (value 0 represents default).
             self.csi_params.push(0);
         }
         // Now add the slot for the *next* parameter, if space allows.
         if self.csi_params.len() < MAX_CSI_PARAMS {
             self.csi_params.push(0);
         }
         // If MAX_CSI_PARAMS is reached, further semicolons are ignored by the length check.
    }


     /// Gets the nth CSI parameter (0-based index), defaulting if absent/0.
    fn get_csi_param(&self, index: usize, default: u16) -> u16 {
        match self.csi_params.get(index) {
            Some(&p) if p > 0 => p, // Return the parameter if it's non-zero
            _ => default,          // Otherwise, return the default value
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

        // Rotate lines upwards
        screen.rotate_left(lines_to_scroll);

        // Clear the newly exposed lines at the bottom
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
        // Calculate new potential positions, ensuring they don't go below zero
        let new_x = (self.cursor.x as isize + dx).max(0) as usize;
        let new_y = (self.cursor.y as isize + dy).max(0) as usize;
        // Clamp the new positions to the terminal boundaries
        self.cursor.x = min(new_x, width.saturating_sub(1));
        self.cursor.y = min(new_y, height.saturating_sub(1));
    }


    /// Sets the cursor to an absolute position (0-based), clamping to bounds.
    fn set_cursor_pos(&mut self, x: usize, y: usize) {
        // Read dimensions before modifying cursor
        let width = self.width;
        let height = self.height;
        // Clamp the new positions to the terminal boundaries
        self.cursor.x = min(x, width.saturating_sub(1));
        self.cursor.y = min(y, height.saturating_sub(1));
    }


    /// Handles Line Feed (LF). Moves cursor down, scrolls if at bottom.
    fn newline(&mut self) {
        let height = self.height; // Read height before potentially calling scroll_up (which needs &mut self)
        let next_y = self.cursor.y + 1;
        if next_y >= height {
            // If at the bottom, scroll up
            self.scroll_up(1);
        } else {
            // Otherwise, just move the cursor down
            self.cursor.y = next_y;
        }
        // Always move to the first column on a newline
        self.cursor.x = 0;
    }


    /// Handles Carriage Return (CR). Moves cursor to the beginning of the line.
    fn carriage_return(&mut self) {
        self.cursor.x = 0;
    }

    /// Handles Backspace (BS). Moves cursor left, stopping at the first column.
    fn backspace(&mut self) {
        if self.cursor.x > 0 {
            self.cursor.x -= 1;
        }
    }

    /// Handles Horizontal Tab (HT). Moves cursor to the next tab stop.
    fn tab(&mut self) {
        // Read immutable fields first
        let current_x = self.cursor.x;
        let width = self.width;
        let tabs = &self.tabs; // Borrow immutably

        // Find the index of the next tab stop after the current position
        let next_tab_pos = tabs.iter()
            .skip(current_x + 1) // Start searching from the next column
            .position(|&is_stop| is_stop); // Find the first true value

        match next_tab_pos {
            Some(relative_pos) => {
                // Calculate absolute position and clamp to width
                let absolute_pos = current_x + 1 + relative_pos;
                self.cursor.x = min(absolute_pos, width.saturating_sub(1));
            }
            None => {
                // No more tab stops, move to the last column
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
        // Create a glyph with a space and the *current* drawing attributes
        let fill_glyph = Glyph { c: ' ', attr: self.current_attributes };

        // Check bounds before mutable borrow
        if y >= height || x_start >= x_end || x_start >= width {
            return; // Invalid range or out of bounds
        }

        // Now borrow screen mutably
        let screen = self.current_screen_mut();
        // Ensure the end range doesn't exceed the screen width
        let clamped_end = min(x_end, width);
        // Fill the specified range on the given line
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
        // Fill from current cursor position to the end of the line
        self.fill_range(y, x, width);
    }

    /// Erases from start of line to cursor (inclusive) (EL 1).
    fn erase_line_to_start(&mut self) {
        // Read immutable fields needed by fill_range
        let y = self.cursor.y;
        let x = self.cursor.x;
        // Fill from the beginning of the line up to and including the cursor
        self.fill_range(y, 0, x + 1);
    }

    /// Erases the entire current line (EL 2).
    fn erase_whole_line(&mut self) {
        // Read immutable fields needed by fill_range
        let y = self.cursor.y;
        let width = self.width;
        // Fill the entire line
        self.fill_range(y, 0, width);
    }

    /// Erases from cursor to end of display (ED 0).
    fn erase_display_to_end(&mut self) {
        // Read immutable fields first
        let cursor_y = self.cursor.y;
        let height = self.height;
        // Create a fill glyph based on current attributes
        let fill_glyph = Glyph { c: ' ', attr: self.current_attributes };

        // Erase rest of current line first
        self.erase_line_to_end();

        // Now borrow screen mutably for lines below
        let screen = self.current_screen_mut();
        // Fill all lines below the cursor
        for y in (cursor_y + 1)..height {
            screen[y].fill(fill_glyph);
        }
    }


    /// Erases from start of display to cursor (inclusive) (ED 1).
    fn erase_display_to_start(&mut self) {
        // Read immutable fields first
        let cursor_y = self.cursor.y;
        let height = self.height; // Only needed for loop bound below
        // Create a fill glyph based on current attributes
        let fill_glyph = Glyph { c: ' ', attr: self.current_attributes };

        // Erase beginning of current line (inclusive of cursor)
        self.erase_line_to_start();

        // Now borrow screen mutably for lines above
        let screen = self.current_screen_mut();
        // Fill all lines above the cursor line
        for y in 0..cursor_y {
             if y < height { // Ensure y is within bounds (redundant due to loop condition)
                 screen[y].fill(fill_glyph);
             }
        }
    }


    /// Erases the entire display (ED 2).
    fn erase_whole_display(&mut self) {
        // Read immutable fields first
        let height = self.height;
        // Create a fill glyph based on current attributes
        let fill_glyph = Glyph { c: ' ', attr: self.current_attributes };

        // Now borrow screen mutably
        let screen = self.current_screen_mut();
        // Fill every line
        for y in 0..height {
            screen[y].fill(fill_glyph);
        }
        // Optionally move cursor home, though not strictly required by ED 2
        // self.set_cursor_pos(0, 0);
    }



    // --- Character Handling ---
    /// Handles a printable character decoded from the input stream.
    fn handle_printable(&mut self, c: char) {
        // TODO: Implement proper Unicode width calculation (e.g., using unicode-width crate)
        let char_width = 1; // Assume width 1 for now
        let width = self.width; // Read width before potential mutable borrow in newline
        let height = self.height; // Read height before potential mutable borrow
        let current_attributes = self.current_attributes; // Copy attributes for the new glyph

        // Check wrap condition using local width
        if self.cursor.x + char_width > width {
            // If we are not already past the end (which shouldn't happen with proper clamping)
            // and wrap mode is enabled, move to the next line.
             if self.cursor.x < width { // Check we are within bounds before wrap
                 self.newline(); // Needs &mut self
             } else {
                 // If somehow cursor is already past the end, just move to next line start.
                 // This case indicates a potential logic error elsewhere.
                 self.newline(); // Needs &mut self
             }
        }

        // Check bounds again using local height/width *after* potential newline
        if self.cursor.y < height && self.cursor.x < width {
            let y = self.cursor.y;
            let x = self.cursor.x;
            // Borrow screen mutably *only* for the write
            let screen = self.current_screen_mut();
            // Place the character with current attributes
            screen[y][x] = Glyph { c, attr: current_attributes };

            // Advance cursor only if character has positive width
            if char_width > 0 {
                self.cursor.x += char_width;
            }
        }
    }



    // --- Escape Sequence Parsing States ---

    /// Processes a single byte received from the PTY or input source.
    pub fn process_byte(&mut self, byte: u8) {
        // Delegate byte handling based on the current parser state
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
            // C0 Control Codes
            0x08 => self.backspace(),       // BS
            0x09 => self.tab(),             // HT
            0x0A | 0x0B | 0x0C => self.newline(), // LF, VT, FF
            0x0D => self.carriage_return(), // CR
            0x00..=0x07 | 0x0E..=0x1A | 0x1C..=0x1F => { /* Ignore other C0 */ }
            0x1B => { // ESC
                self.parser_state = ParserState::Escape;
                self.clear_csi_params(); // Clear params for the new sequence
            }
            // Printable characters (including UTF-8 sequences)
            _ => {
                match self.utf8_decoder.decode(byte) {
                    Ok(Some(c)) => self.handle_printable(c), // Valid char decoded
                    Ok(None) => { /* Need more bytes for UTF-8 */ }
                    Err(_) => { // Invalid UTF-8 sequence
                        self.handle_printable(REPLACEMENT_CHARACTER);
                        // Reset decoder is handled internally by decode on error
                    }
                }
            }
        }
    }


    /// Handles bytes after an ESC (0x1B) has been received.
    fn handle_escape_byte(&mut self, byte: u8) {
        match byte {
            b'[' => self.parser_state = ParserState::CSIEntry, // Enter CSI state
            b']' => self.parser_state = ParserState::OSCParse, // Enter OSC state
            b'c' => { // RIS - Reset to Initial State
                // Read default attributes before mutable borrow
                let default_glyph = Glyph { c: ' ', attr: self.default_attributes };
                // Clear the current screen
                let screen = self.current_screen_mut();
                for row in screen.iter_mut() { row.fill(default_glyph); }
                // Reset cursor, attributes, modes
                self.cursor = Cursor { x: 0, y: 0 };
                self.current_attributes = self.default_attributes;
                self.dec_modes = DecModes::default();
                // Reset parser state
                self.parser_state = ParserState::Ground;
            }
            b'7' => { // DECSC - Save Cursor
                self.saved_cursor = self.cursor;
                self.parser_state = ParserState::Ground;
            }
            b'8' => { // DECRC - Restore Cursor
                self.cursor = self.saved_cursor;
                self.parser_state = ParserState::Ground;
            }
            // Ignore Shift/Character Set selections for now
            b'(' | b')' | b'*' | b'+' => self.parser_state = ParserState::Ground, // Ignore SCS
            // Ignore keypad modes (handled by CSI ? 1 h/l)
            b'=' | b'>' => self.parser_state = ParserState::Ground,
            // Unhandled ESC sequences return to Ground state
            _ => self.parser_state = ParserState::Ground,
        }
    }


    /// Handles the first byte after ESC [.
    fn handle_csi_entry_byte(&mut self, byte: u8) {
        match byte {
            // Parameter bytes
            b'0'..=b'9' => {
                self.push_csi_param((byte - b'0') as u16);
                self.parser_state = ParserState::CSIParam;
            }
            // Parameter separator
            b';' => {
                self.next_csi_param();
                self.parser_state = ParserState::CSIParam;
            }
            // Private marker or intermediate bytes
            b'?' | b'>' | b'!' | b':' => { // Note: ':' is less common but sometimes used
                if self.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                     self.csi_intermediates.push(byte as char);
                     self.parser_state = ParserState::CSIIntermediate;
                 } else {
                     // Too many intermediates, ignore the rest of the sequence
                     self.parser_state = ParserState::CSIIgnore;
                 }
            }
            // Final byte (dispatch the command)
            0x40..=0x7E => {
                self.csi_dispatch(byte);
                self.parser_state = ParserState::Ground;
            }
            // Invalid CSI entry byte, return to Ground
            _ => self.parser_state = ParserState::Ground,
        }
    }


    /// Handles CSI parameter bytes (digits, ';').
    fn handle_csi_param_byte(&mut self, byte: u8) {
        match byte {
            // Parameter digit
            b'0'..=b'9' => self.push_csi_param((byte - b'0') as u16),
            // Parameter separator
            b';' => self.next_csi_param(),
            // Intermediate bytes (can appear after parameters)
            0x20..=0x2F => {
                 if self.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                     self.csi_intermediates.push(byte as char);
                     self.parser_state = ParserState::CSIIntermediate;
                 } else {
                     // Too many intermediates, ignore the rest
                     self.parser_state = ParserState::CSIIgnore;
                 }
            }
            // Final byte (dispatch the command)
            0x40..=0x7E => {
                self.csi_dispatch(byte);
                self.parser_state = ParserState::Ground;
            }
            // Invalid CSI parameter byte, return to Ground
            _ => self.parser_state = ParserState::Ground,
        }
    }


    /// Handles CSI intermediate bytes (after params, before final byte).
    fn handle_csi_intermediate_byte(&mut self, byte: u8) {
        match byte {
            // Collect intermediate byte
            0x20..=0x2F => {
                if self.csi_intermediates.len() < MAX_CSI_INTERMEDIATES {
                    self.csi_intermediates.push(byte as char);
                    // Stay in CSIIntermediate state to collect more
                } else {
                    // Too many intermediates, ignore the rest
                    self.parser_state = ParserState::CSIIgnore;
                }
            }
            // Final byte (dispatch the command)
            0x40..=0x7E => {
                self.csi_dispatch(byte);
                self.parser_state = ParserState::Ground;
            }
            // Handle parameters after intermediates (like '?')
            b'0'..=b'9' => {
                self.push_csi_param((byte - b'0') as u16);
                self.parser_state = ParserState::CSIParam;
            }
            b';' => {
                self.next_csi_param();
                self.parser_state = ParserState::CSIParam;
            }
            // Invalid intermediate byte, return to Ground
            _ => self.parser_state = ParserState::Ground,
        }
    }


    /// Handles bytes when ignoring the rest of a CSI sequence.
    fn handle_csi_ignore_byte(&mut self, byte: u8) {
        // Wait for a final byte (0x40-0x7E) to end the ignored sequence
        if (0x40..=0x7E).contains(&byte) {
            self.parser_state = ParserState::Ground;
        }
        // Otherwise, stay in CSIIgnore state
    }

    /// Handles bytes during OSC sequence parsing (ESC ] ... ST/BEL).
    fn handle_osc_parse_byte(&mut self, byte: u8) {
        // TODO: Implement proper OSC string collection and handling
        // For now, just wait for a terminator.
        match byte {
            0x07 | 0x1B => { // BEL or ESC terminates OSC sequence
                // If ESC, check next byte for ST ('\')
                // For simplicity now, assume ESC alone terminates or is start of new seq.
                // TODO: Handle ESC \ sequence terminator correctly.
                self.parser_state = ParserState::Ground;
            }
            0x00..=0x1A | 0x1C..=0x1F => {} // Ignore C0 control codes within OSC
            _ => {
                // TODO: Collect byte into an OSC buffer if implementing OSC handling.
                // Need to handle buffer limits.
            }
        }
    }



    /// Dispatches action based on the final byte of a CSI sequence.
    fn csi_dispatch(&mut self, final_byte: u8) {
        // Check if it's a private sequence (starts with '?')
        let is_private = self.csi_intermediates.first() == Some(&'?');

        match final_byte {
            // Cursor Movement
            b'A' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(0, -n); } // CUU - Up
            b'B' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(0, n); }  // CUD - Down
            b'C' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(n, 0); }  // CUF - Forward
            b'D' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(-n, 0); } // CUB - Backward
            b'E' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(0, n); self.cursor.x = 0; } // CNL - Next Line
            b'F' => { let n = self.get_csi_param(0, 1) as isize; self.move_cursor(0, -n); self.cursor.x = 0; } // CPL - Previous Line
            b'G' => { let n = self.get_csi_param(0, 1); self.cursor.x = min(n.saturating_sub(1) as usize, self.width.saturating_sub(1)); } // CHA - Horizontal Absolute
            b'H' | b'f' => { // CUP / HVP - Position
                let y = self.get_csi_param(0, 1); // Row (1-based)
                let x = self.get_csi_param(1, 1); // Column (1-based)
                self.set_cursor_pos(x.saturating_sub(1) as usize, y.saturating_sub(1) as usize);
            }

            // Erasing
            b'J' => { // ED - Erase Display
                match self.get_csi_param_or_0(0) {
                    0 => self.erase_display_to_end(),
                    1 => self.erase_display_to_start(),
                    2 => self.erase_whole_display(),
                    _ => {} // Ignore invalid parameters
                }
            }
            b'K' => { // EL - Erase Line
                match self.get_csi_param_or_0(0) {
                    0 => self.erase_line_to_end(),
                    1 => self.erase_line_to_start(),
                    2 => self.erase_whole_line(),
                    _ => {} // Ignore invalid parameters
                }
            }

            // Attributes
            b'm' => self.handle_sgr(), // SGR - Select Graphic Rendition

            // Cursor Save/Restore
            b's' => self.saved_cursor = self.cursor, // DECSC (often) / SCOSC (ANSI)
            b'u' => self.cursor = self.saved_cursor, // DECRC (often) / SCORC (ANSI)

            // Mode Setting
            b'h' if is_private => self.handle_dec_mode_set(true),  // DECSET
            b'l' if is_private => self.handle_dec_mode_set(false), // DECRST

            // Special handling for Alt Screen Buffer (DECSET/DECRST 1049)
            b'h' if is_private && self.get_csi_param_or_0(0) == 1049 => {
                 if !self.using_alt_screen {
                     mem::swap(&mut self.screen, &mut self.alt_screen);
                     self.using_alt_screen = true;
                     self.erase_whole_display(); // Clear the alt screen
                     self.set_cursor_pos(0, 0);   // Move cursor to home
                 }
             }
             b'l' if is_private && self.get_csi_param_or_0(0) == 1049 => {
                 if self.using_alt_screen {
                     mem::swap(&mut self.screen, &mut self.alt_screen);
                     self.using_alt_screen = false;
                     // Screen content is restored, cursor position usually restored by DECRC
                 }
             }

            // TODO: Add more CSI commands as needed (Scrolling, Insert/Delete Chars/Lines, etc.)

            _ => { /* Ignore unknown CSI sequences */ }
        }
    }


    /// Handles SGR (Select Graphic Rendition) codes (CSI ... m).
    fn handle_sgr(&mut self) {
        // If no parameters or only 0, reset all attributes
        if self.csi_params.is_empty() || self.get_csi_param_or_0(0) == 0 {
            self.current_attributes = self.default_attributes;
            // If there were other parameters after 0, process them
            if self.csi_params.len() > 1 {
                self.process_sgr_codes(1); // Start processing from the second parameter
            }
            return;
        }
        // Process all parameters starting from the first one
        self.process_sgr_codes(0);
    }

    /// Processes SGR codes starting from a given index in `csi_params`.
    fn process_sgr_codes(&mut self, start_index: usize) {
        let mut i = start_index;
        while i < self.csi_params.len() {
            let code = self.get_csi_param_or_0(i);
            match code {
                // Reset / Normal
                0 => self.current_attributes = self.default_attributes,

                // Intensity / Style Flags
                1 => self.current_attributes.flags |= AttrFlags::BOLD,
                3 => self.current_attributes.flags |= AttrFlags::ITALIC,
                4 => self.current_attributes.flags |= AttrFlags::UNDERLINE,
                7 => self.current_attributes.flags |= AttrFlags::REVERSE,
                8 => self.current_attributes.flags |= AttrFlags::HIDDEN,
                9 => self.current_attributes.flags |= AttrFlags::STRIKETHROUGH,

                // Reset Intensity / Style Flags
                22 => self.current_attributes.flags &= !AttrFlags::BOLD, // Normal intensity
                23 => self.current_attributes.flags &= !AttrFlags::ITALIC,
                24 => self.current_attributes.flags &= !AttrFlags::UNDERLINE,
                27 => self.current_attributes.flags &= !AttrFlags::REVERSE,
                28 => self.current_attributes.flags &= !AttrFlags::HIDDEN,
                29 => self.current_attributes.flags &= !AttrFlags::STRIKETHROUGH,

                // Basic Foreground Colors (30-37)
                30..=37 => self.current_attributes.fg = Color::Idx((code - 30) as u8),
                // Extended Foreground Colors (38)
                38 => {
                    i += 1; // Move to the next parameter (type specifier)
                    match self.get_csi_param_or_0(i) {
                        5 => { // 256-color mode
                            i += 1; // Move to the color index parameter
                            let idx = self.get_csi_param_or_0(i) as u8;
                            self.current_attributes.fg = Color::Idx(idx);
                        }
                        2 => { // Truecolor (RGB) mode
                            // Check if we have enough parameters for R, G, B
                            if i + 3 < self.csi_params.len() {
                                let r = self.get_csi_param_or_0(i + 1) as u8;
                                let g = self.get_csi_param_or_0(i + 2) as u8;
                                let b = self.get_csi_param_or_0(i + 3) as u8;
                                self.current_attributes.fg = Color::Rgb(r, g, b);
                                i += 3; // Consume R, G, B parameters
                            } else {
                                // Not enough parameters for RGB, stop processing SGR
                                i = self.csi_params.len();
                            }
                        }
                        _ => {
                             // Invalid or unsupported type specifier for 38
                             // Skip the type specifier if it wasn't 5 or 2
                             // If there was another value (like color index for 5),
                             // we might need to skip that too, depending on exact spec.
                             // For simplicity, we might just stop processing SGR here.
                             // Let's skip one potential value parameter for now.
                             i += 1;
                        }
                    }
                }
                // Default Foreground Color (39)
                39 => self.current_attributes.fg = self.default_attributes.fg,

                // Basic Background Colors (40-47)
                40..=47 => self.current_attributes.bg = Color::Idx((code - 40) as u8),
                // Extended Background Colors (48)
                48 => {
                    i += 1; // Move to the type specifier
                    match self.get_csi_param_or_0(i) {
                        5 => { // 256-color mode
                            i += 1; // Move to the color index
                            let idx = self.get_csi_param_or_0(i) as u8;
                            self.current_attributes.bg = Color::Idx(idx);
                        }
                        2 => { // Truecolor (RGB) mode
                            if i + 3 < self.csi_params.len() {
                                let r = self.get_csi_param_or_0(i + 1) as u8;
                                let g = self.get_csi_param_or_0(i + 2) as u8;
                                let b = self.get_csi_param_or_0(i + 3) as u8;
                                self.current_attributes.bg = Color::Rgb(r, g, b);
                                i += 3; // Consume R, G, B
                            } else {
                                i = self.csi_params.len(); // Not enough params
                            }
                        }
                         _ => {
                             // Invalid or unsupported type specifier for 48
                             i += 1; // Skip potential value parameter
                         }
                    }
                }
                // Default Background Color (49)
                49 => self.current_attributes.bg = self.default_attributes.bg,

                // Bright Foreground Colors (90-97)
                90..=97 => self.current_attributes.fg = Color::Idx((code - 90 + 8) as u8),
                // Bright Background Colors (100-107)
                100..=107 => self.current_attributes.bg = Color::Idx((code - 100 + 8) as u8),

                _ => { /* Ignore unknown SGR codes */ }
            }
            i += 1; // Move to the next parameter code
        }
    }


    /// Handles DEC Private Mode Set/Reset sequences (CSI ? Pn h/l).
    fn handle_dec_mode_set(&mut self, enable: bool) {
        // Iterate through all parameters provided in the sequence
        for i in 0..self.csi_params.len() {
             let mode = self.get_csi_param_or_0(i);
             match mode {
                1 => self.dec_modes.cursor_keys_app_mode = enable, // DECCKM
                // 1049 (Alt screen buffer) is handled directly in csi_dispatch
                // Add cases for other DEC modes here:
                // 7 => // DECAWM (Auto Wrap) - Handled by term.mode flag
                // 25 => // DECTCEM (Text Cursor Enable) - Handled by backend/draw logic
                // 1000, 1002, 1003, 1006 => // Mouse modes - Need state in DecModes or Term
                // 2004 => // Bracketed Paste - Need state in DecModes or Term
                1049 => { /* Handled specially in csi_dispatch */ }
                _ => { /* Ignore unknown DEC private modes */ }
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
        assert_eq!(term.get_glyph(3, 0).unwrap().c, ' '); // Default char
    }

    #[test]
    fn test_line_wrap() {
        let term = term_with_bytes(3, 2, b"abcde");
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 1);
        assert_eq!(term.screen_to_string_line(0), "abc");
        assert_eq!(term.screen_to_string_line(1), "de "); // Space from default glyph
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
        assert_eq!(term.screen_to_string_line(0), "dec  "); // d overwrites a, e overwrites b
    }

    #[test]
    fn test_backspace() {
        let term = term_with_bytes(5, 1, b"abc\x08d");
        // Cursor moves from 3 to 2 after BS, then 'd' overwrites 'c' at pos 2, cursor moves to 3
        assert_eq!(term.cursor.x, 3);
        assert_eq!(term.screen_to_string_line(0), "abd  ");
    }

    // --- Parser State Tests ---
    #[test]
    fn test_parser_state_csi() {
        let mut term = Term::new(10, 5);
        term.process_byte(0x1b); // ESC
        assert_eq!(term.parser_state, ParserState::Escape);
        term.process_byte(b'['); // CSI Entry
        assert_eq!(term.parser_state, ParserState::CSIEntry);
        term.process_byte(b'1'); // Param digit
        assert_eq!(term.parser_state, ParserState::CSIParam);
        term.process_byte(b';'); // Param separator
        assert_eq!(term.parser_state, ParserState::CSIParam);
        term.process_byte(b'2'); // Param digit
        assert_eq!(term.parser_state, ParserState::CSIParam);
        term.process_byte(b'H'); // Final byte (CUP)
        assert_eq!(term.parser_state, ParserState::Ground); // Back to ground
        // CUP 1;2 -> row 1, col 2 (0-indexed 0, 1)
        assert_eq!(term.cursor.x, 1);
        assert_eq!(term.cursor.y, 0);
    }

    #[test]
    fn test_esc_keypad_modes() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b="); // DECKPAM
        assert_eq!(term.parser_state, ParserState::Ground);
        term.process_bytes(b"\x1b>"); // DECKPNM
        assert_eq!(term.parser_state, ParserState::Ground);
    }

    #[test]
    fn test_esc_intermediate_charset() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b(B"); // Select G0 charset B (USASCII)
        assert_eq!(term.parser_state, ParserState::Ground);
    }

    // --- CSI Cursor Movement Tests ---
    #[test]
    fn test_csi_cursor_up() {
        let mut term = Term::new(10, 5);
        term.set_cursor_pos(3, 3);
        term.process_bytes(b"\x1b[A"); // CUU 1
        assert_eq!(term.cursor, Cursor { x: 3, y: 2 });
    }

    #[test]
    fn test_csi_cursor_up_param() {
        let mut term = Term::new(10, 5);
        term.set_cursor_pos(3, 3);
        term.process_bytes(b"\x1b[2A"); // CUU 2
        assert_eq!(term.cursor, Cursor { x: 3, y: 1 });
    }

    #[test]
    fn test_csi_cursor_down() {
        let mut term = Term::new(10, 5);
        term.set_cursor_pos(3, 1);
        term.process_bytes(b"\x1b[B"); // CUD 1
        assert_eq!(term.cursor, Cursor { x: 3, y: 2 });
        term.process_bytes(b"\x1b[0B"); // CUD 1 (0 defaults to 1)
        assert_eq!(term.cursor, Cursor { x: 3, y: 3 });
        term.process_bytes(b"\x1b[1B"); // CUD 1
        assert_eq!(term.cursor, Cursor { x: 3, y: 4 });
    }

    #[test]
    fn test_csi_cursor_forward() {
        let mut term = Term::new(10, 5);
        term.set_cursor_pos(1, 1);
        term.process_bytes(b"\x1b[C"); // CUF 1
        assert_eq!(term.cursor, Cursor { x: 2, y: 1 });
        term.process_bytes(b"\x1b[2C"); // CUF 2
        assert_eq!(term.cursor, Cursor { x: 4, y: 1 });
    }

    #[test]
    fn test_csi_cursor_backward() {
        let mut term = Term::new(10, 5);
        term.set_cursor_pos(3, 1);
        term.process_bytes(b"\x1b[D"); // CUB 1
        assert_eq!(term.cursor, Cursor { x: 2, y: 1 });
        term.process_bytes(b"\x1b[2D"); // CUB 2
        assert_eq!(term.cursor, Cursor { x: 0, y: 1 });
        term.process_bytes(b"\x1b[D"); // CUB 1 (already at 0)
        assert_eq!(term.cursor, Cursor { x: 0, y: 1 }); // Stays at 0
    }

    #[test]
    fn test_csi_cursor_position() {
        let term = term_with_bytes(10, 5, b"\x1b[3;4H"); // CUP row 3, col 4 -> (y=2, x=3)
        assert_eq!(term.cursor, Cursor { x: 3, y: 2 });
        let term2 = term_with_bytes(10, 5, b"\x1b[H"); // CUP default (1;1) -> (y=0, x=0)
        assert_eq!(term2.cursor, Cursor { x: 0, y: 0 });
        let term3 = term_with_bytes(10, 5, b"\x1b[;5H"); // CUP default row 1, col 5 -> (y=0, x=4)
        assert_eq!(term3.cursor, Cursor { x: 4, y: 0 });
        let term4 = term_with_bytes(10, 5, b"\x1b[5;H"); // CUP row 5, default col 1 -> (y=4, x=0)
        assert_eq!(term4.cursor, Cursor { x: 0, y: 4 });
    }

    // --- CSI Erase Tests ---
    #[test]
    fn test_csi_erase_line_to_end() {
        let mut term = Term::new(5, 1);
        term.process_bytes(b"abcde");
        term.set_cursor_pos(2, 0); // Cursor at 'c'
        term.process_bytes(b"\x1b[K"); // EL 0
        assert_eq!(term.screen_to_string_line(0), "ab   ");
        assert_eq!(term.cursor.x, 2); // Cursor doesn't move
    }

    #[test]
    fn test_csi_erase_line_to_start() {
        let mut term = Term::new(5, 1);
        term.process_bytes(b"abcde");
        term.set_cursor_pos(2, 0); // Cursor at 'c'
        term.process_bytes(b"\x1b[1K"); // EL 1
        assert_eq!(term.screen_to_string_line(0), "   de"); // a, b, c erased
        assert_eq!(term.cursor.x, 2); // Cursor doesn't move
    }

    #[test]
    fn test_csi_erase_whole_line() {
        let mut term = Term::new(5, 1);
        term.process_bytes(b"abcde");
        term.set_cursor_pos(2, 0); // Cursor at 'c'
        term.process_bytes(b"\x1b[2K"); // EL 2
        assert_eq!(term.screen_to_string_line(0), "     "); // Whole line erased
        assert_eq!(term.cursor.x, 2); // Cursor doesn't move
    }

    // --- DEC Private Mode Tests ---
    #[test]
    fn test_dec_mode_set_reset() {
        let mut term = Term::new(10, 5);
        assert!(!term.dec_modes.cursor_keys_app_mode);
        term.process_bytes(b"\x1b[?1h"); // DECSET mode 1 (DECCKM)
        assert!(term.dec_modes.cursor_keys_app_mode);
        term.process_bytes(b"\x1b[?1l"); // DECRST mode 1 (DECCKM)
        assert!(!term.dec_modes.cursor_keys_app_mode);
    }

    // --- SGR Tests ---
    #[test]
    fn test_sgr_reset() {
        let mut term = Term::new(10, 5);
        term.current_attributes.fg = Color::Idx(1); // Red
        term.current_attributes.flags |= AttrFlags::BOLD;
        term.process_bytes(b"\x1b[m"); // SGR 0 (implicit)
        assert_eq!(term.current_attributes, term.default_attributes);
        term.current_attributes.fg = Color::Idx(1); // Red again
        term.process_bytes(b"\x1b[0m"); // SGR 0 (explicit)
        assert_eq!(term.current_attributes, term.default_attributes);
    }

    #[test]
    fn test_sgr_basic_colors() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[31m"); // FG Red
        assert_eq!(term.current_attributes.fg, Color::Idx(1));
        term.process_bytes(b"\x1b[42m"); // BG Green
        assert_eq!(term.current_attributes.bg, Color::Idx(2));
        term.process_bytes(b"\x1b[39m"); // Default FG
        assert_eq!(term.current_attributes.fg, Color::Default);
        term.process_bytes(b"\x1b[49m"); // Default BG
        assert_eq!(term.current_attributes.bg, Color::Default);
    }

    #[test]
    fn test_sgr_bright_colors() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[91m"); // Bright FG Red
        assert_eq!(term.current_attributes.fg, Color::Idx(9)); // 1 + 8
        term.process_bytes(b"\x1b[102m"); // Bright BG Green
        assert_eq!(term.current_attributes.bg, Color::Idx(10)); // 2 + 8
    }

    #[test]
    fn test_sgr_bold_underline() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[1m"); // Bold on
        assert!(term.current_attributes.flags.contains(AttrFlags::BOLD));
        term.process_bytes(b"\x1b[4m"); // Underline on
        assert!(term.current_attributes.flags.contains(AttrFlags::UNDERLINE));
        assert!(term.current_attributes.flags.contains(AttrFlags::BOLD)); // Bold still on
        term.process_bytes(b"\x1b[22m"); // Bold off (Normal intensity)
        assert!(!term.current_attributes.flags.contains(AttrFlags::BOLD));
        assert!(term.current_attributes.flags.contains(AttrFlags::UNDERLINE)); // Underline still on
        term.process_bytes(b"\x1b[24m"); // Underline off
        assert!(!term.current_attributes.flags.contains(AttrFlags::UNDERLINE));
    }

    #[test]
    fn test_sgr_reverse() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[7m"); // Reverse on
        assert!(term.current_attributes.flags.contains(AttrFlags::REVERSE));
        term.process_bytes(b"\x1b[27m"); // Reverse off
        assert!(!term.current_attributes.flags.contains(AttrFlags::REVERSE));
    }

    #[test]
    fn test_sgr_combined() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[1;31;44m"); // Bold, FG Red, BG Blue
        assert!(term.current_attributes.flags.contains(AttrFlags::BOLD));
        assert_eq!(term.current_attributes.fg, Color::Idx(1));
        assert_eq!(term.current_attributes.bg, Color::Idx(4));
        term.process_bytes(b"\x1b[0m"); // Reset all
        assert_eq!(term.current_attributes, term.default_attributes);
    }

    #[test]
    fn test_sgr_256color() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[38;5;208m"); // FG Orange (index 208)
        assert_eq!(term.current_attributes.fg, Color::Idx(208));
        term.process_bytes(b"\x1b[48;5;75m"); // BG Sky Blue (index 75)
        assert_eq!(term.current_attributes.bg, Color::Idx(75));
    }

    #[test]
    fn test_sgr_truecolor() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"\x1b[38;2;10;20;30m"); // FG RGB(10, 20, 30)
        assert_eq!(term.current_attributes.fg, Color::Rgb(10, 20, 30));
        term.process_bytes(b"\x1b[48;2;100;150;200m"); // BG RGB(100, 150, 200)
        assert_eq!(term.current_attributes.bg, Color::Rgb(100, 150, 200));
    }

    // --- UTF-8 Tests ---
    #[test]
    fn test_utf8_basic() {
        let term = term_with_bytes(10, 1, "你好".as_bytes());
        assert_eq!(term.cursor.x, 2); // Assuming width 1 for each CJK char for now
        assert_eq!(term.get_glyph(0, 0).unwrap().c, '你');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, '好');
    }

    #[test]
    fn test_utf8_mixed() {
        let term = term_with_bytes(10, 1, "a你b好c".as_bytes());
        assert_eq!(term.cursor.x, 5); // a(1) + 你(1) + b(1) + 好(1) + c(1) = 5
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'a');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, '你');
        assert_eq!(term.get_glyph(2, 0).unwrap().c, 'b');
        assert_eq!(term.get_glyph(3, 0).unwrap().c, '好');
        assert_eq!(term.get_glyph(4, 0).unwrap().c, 'c');
    }

    #[test]
    fn test_utf8_invalid_sequence() {
        // 0x80 is an invalid continuation byte without a start byte
        let term = term_with_bytes(10, 1, &[0x61, 0x80, 0x62]);
        assert_eq!(term.cursor.x, 3); // a(1) + (1) + b(1) = 3
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'a');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, REPLACEMENT_CHARACTER);
        assert_eq!(term.get_glyph(2, 0).unwrap().c, 'b');
    }

    #[test]
    fn test_utf8_incomplete_sequence_at_end() {
        // 0xE4 needs two more bytes
        let term = term_with_bytes(10, 1, &[0x61, 0xE4]);
        assert_eq!(term.cursor.x, 1); // Only 'a' is processed
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'a');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, ' '); // Rest is default
        // Check decoder state
        assert_eq!(term.utf8_decoder.len, 1);
        assert_eq!(term.utf8_decoder.buffer[0], 0xE4);
    }

    #[test]
    fn test_utf8_incomplete_sequence_then_continue() {
        let mut term = Term::new(10, 1);
        term.process_byte(0x61); // 'a'
        term.process_byte(0xE4); // Start of '你'
        assert_eq!(term.cursor.x, 1);
        assert_eq!(term.utf8_decoder.len, 1);
        term.process_byte(0xBD); // Middle of '你'
        assert_eq!(term.cursor.x, 1); // Still incomplete
        assert_eq!(term.utf8_decoder.len, 2);
        term.process_byte(0xA0); // End of '你'
        assert_eq!(term.cursor.x, 2); // '你' processed, cursor advanced
        assert_eq!(term.get_glyph(1, 0).unwrap().c, '你');
        assert_eq!(term.utf8_decoder.len, 0); // Decoder reset
    }

    #[test]
    fn test_utf8_with_csi() {
        // Sequence: a, CSI Forward, 你, 好
        let term = term_with_bytes(10, 1, b"a\x1b[C\xe4\xbd\xa0\xe5\xa5\xbd");
        // a -> x=1
        // \x1b[C -> x=2
        // 你 -> x=3
        // 好 -> x=4
        assert_eq!(term.cursor.x, 4);
        assert_eq!(term.screen_to_string_line(0), "a 你好      ");
    }

    // --- Resize Test ---
    #[test]
    fn test_resize() {
        let mut term = Term::new(10, 5);
        term.process_bytes(b"123\n456"); // Cursor at (3, 1) after '6'
        term.set_cursor_pos(2, 1); // Manually set cursor to (2, 1) ('6')

        // Shrink
        term.resize(5, 3);
        assert_eq!(term.width, 5);
        assert_eq!(term.height, 3);
        assert_eq!(term.screen.len(), 3);
        assert_eq!(term.screen[0].len(), 5);
        assert_eq!(term.screen[1].len(), 5);
        assert_eq!(term.screen[2].len(), 5);
        assert_eq!(term.screen_to_string_line(0), "123  ");
        assert_eq!(term.screen_to_string_line(1), "456  "); // Content preserved
        assert_eq!(term.screen_to_string_line(2), "     "); // New line
        // Cursor clamped
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
        assert_eq!(term.screen_to_string_line(2), "               "); // Old line 2 preserved (was blank)
        assert_eq!(term.screen_to_string_line(3), "               "); // New line 3
        // Cursor position unchanged
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.cursor.y, 1);
    }

}
