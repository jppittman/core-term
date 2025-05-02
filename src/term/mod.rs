// src/term/mod.rs

// Declare submodules (implementations will go in separate files)
mod parser;
mod screen;
// mod state; // Optional module for state-related structs/enums

// Re-export the main Term struct
// pub use core::Term; // If Term is moved to core.rs
// Or define Term struct directly here for now:

use crate::glyph::{Glyph, Attributes, Color, AttrFlags};
use std::str;

// --- Constants ---
/// Default tab stop interval.
const DEFAULT_TAB_INTERVAL: usize = 8;
/// Maximum number of CSI parameters allowed.
const MAX_CSI_PARAMS: usize = 16;
/// Maximum number of CSI intermediate bytes allowed.
const MAX_CSI_INTERMEDIATES: usize = 2;
/// Maximum length for OSC strings.
const MAX_OSC_STRING_LEN: usize = 1024;

// --- Helper Structs/Enums (Moved from term.rs) ---

// Helper struct for UTF-8 decoding
#[derive(Debug)]
struct Utf8Decoder {
    buffer: [u8; 4],
    len: usize,
}

// Represents the cursor position (0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cursor {
    pub x: usize,
    pub y: usize,
}

// States for the terminal escape sequence parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParserState {
    Ground,
    Escape,
    CSIEntry,
    CSIParam,
    CSIIntermediate,
    CSIIgnore,
    OSCParse,
    OSCParam,
}

// Holds DEC Private Mode settings.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DecModes {
    cursor_keys_app_mode: bool,
}

// --- Utf8Decoder Implementation ---
impl Utf8Decoder {
    fn new() -> Self {
        Utf8Decoder { buffer: [0; 4], len: 0 }
    }

    /// Decodes the next byte in a potential UTF-8 sequence.
    fn decode(&mut self, byte: u8) -> Result<Option<char>, ()> {
        if byte.is_ascii() && self.len == 0 {
            return Ok(Some(byte as char));
        }
        if byte.is_ascii() && self.len > 0 {
            self.len = 0;
            return Err(());
        }
        if self.len >= 4 {
            self.len = 0;
            return Err(());
        }
        self.buffer[self.len] = byte;
        self.len += 1;

        match str::from_utf8(&self.buffer[0..self.len]) {
            Ok(s) => {
                let expected_len = parser::utf8_char_len(self.buffer[0]); // Use helper from parser module
                if self.len == expected_len {
                    self.len = 0;
                    Ok(s.chars().next())
                } else if self.len < expected_len {
                    Ok(None)
                } else {
                    self.len = 0;
                    Err(())
                }
            }
            Err(e) => {
                if e.error_len().is_none() && self.len < 4 {
                    Ok(None)
                } else {
                    self.len = 0;
                    Err(())
                }
            }
        }
    }
}

// --- DecModes Implementation ---
impl Default for DecModes {
    fn default() -> Self {
        DecModes {
            cursor_keys_app_mode: false,
        }
    }
}


// --- Main Term Struct Definition ---
pub struct Term {
    width: usize,
    height: usize,
    cursor: Cursor,
    screen: Vec<Vec<Glyph>>,
    alt_screen: Vec<Vec<Glyph>>,
    using_alt_screen: bool,
    saved_cursor: Cursor,
    saved_cursor_alt: Cursor,
    current_attributes: Attributes,
    default_attributes: Attributes,
    parser_state: ParserState,
    csi_params: Vec<u16>,
    csi_intermediates: Vec<char>,
    osc_string: String,
    utf8_decoder: Utf8Decoder,
    dec_modes: DecModes,
    tabs: Vec<bool>,
    // TODO: Implement scrolling region properly
    // top: usize,
    // bot: usize,
}

// --- Public API Implementation ---
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
            saved_cursor_alt: Cursor { x: 0, y: 0 },
            current_attributes: default_attributes,
            default_attributes,
            parser_state: ParserState::Ground,
            csi_params: Vec::with_capacity(MAX_CSI_PARAMS),
            csi_intermediates: Vec::with_capacity(MAX_CSI_INTERMEDIATES),
            osc_string: String::with_capacity(MAX_OSC_STRING_LEN),
            utf8_decoder: Utf8Decoder::new(),
            dec_modes: DecModes::default(),
            tabs,
            // top: 0, // TODO: Uncomment when scrolling region is implemented
            // bot: height - 1, // TODO: Uncomment when scrolling region is implemented
        }
    }

    /// Gets the glyph at the specified coordinates (0-based).
    pub fn get_glyph(&self, x: usize, y: usize) -> Option<&Glyph> {
        let current_screen = if self.using_alt_screen { &self.alt_screen } else { &self.screen };
        current_screen.get(y).and_then(|row| row.get(x))
    }

    /// Gets the current cursor position (col, row).
    pub fn get_cursor(&self) -> (usize, usize) { (self.cursor.x, self.cursor.y) }

    /// Gets the terminal dimensions (width, height) in cells.
    pub fn get_dimensions(&self) -> (usize, usize) { (self.width, self.height) }

    /// Processes a slice of bytes received from the PTY or input source.
    pub fn process_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.process_byte(byte);
        }
    }

    /// Processes a single byte received from the PTY or input source.
    /// Delegates handling based on the current parser state.
    pub fn process_byte(&mut self, byte: u8) {
        // Delegate to the appropriate handler based on state
        // These handlers will live in parser.rs
        match self.parser_state {
            ParserState::Ground => parser::handle_ground_byte(self, byte),
            ParserState::Escape => parser::handle_escape_byte(self, byte),
            ParserState::CSIEntry => parser::handle_csi_entry_byte(self, byte),
            ParserState::CSIParam => parser::handle_csi_param_byte(self, byte),
            ParserState::CSIIntermediate => parser::handle_csi_intermediate_byte(self, byte),
            ParserState::CSIIgnore => parser::handle_csi_ignore_byte(self, byte),
            ParserState::OSCParam => parser::handle_osc_param_byte(self, byte),
            ParserState::OSCParse => parser::handle_osc_parse_byte(self, byte),
        }
    }

     /// Resizes the terminal emulator state.
     /// (Implementation moved to screen.rs)
     pub fn resize(&mut self, new_width: usize, new_height: usize) {
         screen::resize(self, new_width, new_height);
     }

    // --- Private Helper Methods (Moved to submodules) ---

    /// Returns a mutable reference to the currently active screen buffer.
    /// (Moved to screen.rs)
    fn current_screen_mut(&mut self) -> &mut Vec<Vec<Glyph>> {
        screen::current_screen_mut(self)
    }

    /// Clears CSI parameters and intermediates.
    /// (Moved to parser.rs)
    fn clear_csi_params(&mut self) {
        parser::clear_csi_params(self);
    }

    /// Clears OSC string buffer.
    /// (Moved to parser.rs)
    fn clear_osc_string(&mut self) {
        parser::clear_osc_string(self);
    }

    /// Appends a digit to the last CSI parameter being built.
    /// (Moved to parser.rs)
    fn push_csi_param(&mut self, digit: u16) {
        parser::push_csi_param(self, digit);
    }

    /// Moves to the next CSI parameter slot (appends a 0).
    /// (Moved to parser.rs)
    fn next_csi_param(&mut self) {
        parser::next_csi_param(self);
    }

     /// Gets the nth CSI parameter (0-based index), defaulting if absent/0.
     /// (Moved to parser.rs)
    fn get_csi_param(&self, index: usize, default: u16) -> u16 {
        parser::get_csi_param(self, index, default)
    }

     /// Gets the nth CSI parameter (0-based index), returning 0 if absent.
     /// (Moved to parser.rs)
     fn get_csi_param_or_0(&self, index: usize) -> u16 {
        parser::get_csi_param_or_0(self, index)
    }

    // Other private methods like scroll_up, scroll_down, newline, handle_printable,
    // csi_dispatch, handle_sgr etc. will be moved to screen.rs or parser.rs
}

// --- Unit Tests ---
// Tests will be moved to the respective submodules (parser.rs, screen.rs)
#[cfg(test)]
mod tests {
    // We'll add tests in the submodules later
}
