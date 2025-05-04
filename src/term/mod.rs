// src/term/mod.rs

// Declare submodules (implementations will go in separate files)
mod parser;
mod screen;

// Use necessary items from glyph module
use crate::glyph::{Glyph, Attributes, REPLACEMENT_CHARACTER};
use std::str;
use std::cmp::max; // Import max directly
// Add log crate import
use log::{trace, debug, warn}; // Use warn for unexpected conditions

// --- Constants ---
/// Default tab stop interval.
pub(super) const DEFAULT_TAB_INTERVAL: usize = 8; // Made pub(super) for tests
/// Maximum number of CSI parameters allowed.
const MAX_CSI_PARAMS: usize = 16;
/// Maximum number of CSI intermediate bytes allowed.
const MAX_CSI_INTERMEDIATES: usize = 2;
/// Maximum length for OSC strings.
const MAX_OSC_STRING_LEN: usize = 1024;
/// Size of the UTF-8 decoder buffer. Matches max UTF-8 sequence length.
const UTF8_BUFFER_SIZE: usize = 4;


// --- Helper Structs/Enums ---

/// Decodes a stream of bytes into UTF-8 characters using `std::str::from_utf8`.
/// Maintains internal state to handle multi-byte sequences split across calls.
#[derive(Debug, Default)]
pub(super) struct Utf8Decoder {
    buffer: [u8; UTF8_BUFFER_SIZE],
    len: usize,
}

/// Represents the cursor position (0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cursor {
    pub x: usize,
    pub y: usize,
}

/// States for the terminal escape sequence parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ParserState {
    #[default] // Default state
    Ground,
    Escape,
    CSIEntry,
    CSIParam,
    CSIIntermediate,
    CSIIgnore,
    // OSCParse, // Removed unused state
    // OSCParam, // Removed unused state
    OSCString, // Consolidated state for OSC string collection
}

/// Holds DEC Private Mode settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DecModes {
    cursor_keys_app_mode: bool,
    /// DECOM state (false = absolute, true = relative to scroll region)
    origin_mode: bool,
}

// --- Utf8Decoder Implementation ---
impl Utf8Decoder {
    /// Creates a new, empty UTF-8 decoder.
    fn new() -> Self {
        Self::default()
    }

    /// Resets the decoder state.
    fn reset(&mut self) {
        self.len = 0;
    }

    /// Decodes the next byte in a potential UTF-8 sequence using std::str::from_utf8.
    fn decode(&mut self, byte: u8) -> Result<Option<char>, ()> {
        // Handle simple ASCII case first (optimization and handles control codes)
        if self.len == 0 && byte < 0x80 {
            self.reset(); // Consume the byte
            if byte < 0x20 || byte == parser::DEL {
                return Ok(None); // Signal control code
            } else {
                return Ok(Some(byte as char)); // Printable ASCII
            }
        }

        // Check for buffer overflow before adding
        if self.len >= UTF8_BUFFER_SIZE {
            warn!("UTF-8 decoder buffer overflow, resetting.");
            self.reset();
            // We need to process the current byte *after* resetting
            // Re-call decode with the current byte, now that buffer is empty
            return self.decode(byte);
            // return Err(()); // Error: overflow - previous logic was flawed
        }

        // Add byte to buffer
        self.buffer[self.len] = byte;
        self.len += 1;

        // Try decoding the current buffer content
        match str::from_utf8(&self.buffer[0..self.len]) {
            Ok(s) => {
                // Successfully decoded a character.
                let c = s.chars().next(); // Should always be Some(char) here
                self.reset(); // Clear buffer as character is complete
                Ok(c)
            }
            Err(e) => {
                // Check if the error indicates an incomplete sequence or an invalid byte
                if e.error_len().is_none() && self.len < UTF8_BUFFER_SIZE {
                    // Valid start, but needs more bytes and buffer is not full
                    Ok(None)
                } else {
                    // Invalid sequence (invalid byte, too long, etc.)
                    // Don't warn here, let the caller handle the Err and warn/print replacement
                    self.reset();
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
            origin_mode: false,
        }
    }
}


// --- Main Term Struct Definition ---
pub struct Term {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) cursor: Cursor,
    pub(super) screen: Vec<Vec<Glyph>>,
    pub(super) alt_screen: Vec<Vec<Glyph>>,
    pub(super) using_alt_screen: bool,
    pub(super) saved_cursor: Cursor,
    #[allow(dead_code)] pub(super) saved_attributes: Attributes,
    pub(super) saved_cursor_alt: Cursor,
    #[allow(dead_code)] pub(super) saved_attributes_alt: Attributes,
    pub(super) current_attributes: Attributes,
    pub(super) default_attributes: Attributes,
    pub(super) parser_state: ParserState,
    pub(super) csi_params: Vec<u16>,
    pub(super) csi_intermediates: Vec<char>,
    pub(super) osc_string: String,
    /// UTF-8 decoder state.
    pub(super) utf8_decoder: Utf8Decoder,
    // Modes
    pub(super) dec_modes: DecModes,
    pub(super) tabs: Vec<bool>,
    pub(super) scroll_top: usize,
    pub(super) scroll_bot: usize,
    #[allow(dead_code)] pub(super) dirty: Vec<u8>,
}

// --- Public API Implementation ---
impl Term {
    /// Creates a new terminal state with given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        let default_attributes = Attributes::default();
        let default_glyph = Glyph { c: ' ', attr: default_attributes };
        let width = max(1, width);
        let height = max(1, height);

        let screen = vec![vec![default_glyph; width]; height];
        let alt_screen = vec![vec![default_glyph; width]; height];
        let tabs = (0..width).map(|i| i % DEFAULT_TAB_INTERVAL == 0).collect();
        let dirty = vec![1u8; height];

        Term {
            width, height, screen, alt_screen, using_alt_screen: false,
            cursor: Cursor::default(),
            saved_cursor: Cursor::default(),
            saved_attributes: default_attributes,
            saved_cursor_alt: Cursor::default(),
            saved_attributes_alt: default_attributes,
            current_attributes: default_attributes,
            default_attributes,
            parser_state: ParserState::default(),
            csi_params: Vec::with_capacity(MAX_CSI_PARAMS),
            csi_intermediates: Vec::with_capacity(MAX_CSI_INTERMEDIATES),
            osc_string: String::with_capacity(MAX_OSC_STRING_LEN),
            utf8_decoder: Utf8Decoder::new(),
            dec_modes: DecModes::default(),
            tabs,
            scroll_top: 0,
            scroll_bot: height.saturating_sub(1),
            dirty,
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
        if log::log_enabled!(log::Level::Debug) {
            let printable_bytes: String = bytes.iter().map(|&b| {
                if b.is_ascii_graphic() || b == b' ' { b as char }
                else if b == parser::ESC { '␛' }
                else if b < 0x20 || b == parser::DEL { '.' }
                else { '?' }
            }).collect();
            debug!("Processing {} bytes from PTY: '{}'", bytes.len(), printable_bytes);
        }
        for &byte in bytes {
            self.process_byte(byte);
        }
    }

    /// Processes a single byte received from the PTY or input source.
    pub fn process_byte(&mut self, byte: u8) {
        let initial_state = self.parser_state;
        trace!("process_byte: byte=0x{:02X}, state={:?}", byte, initial_state);

        // If not in Ground state, pass byte directly to the parser state machine
        if initial_state != ParserState::Ground {
             if self.utf8_decoder.len > 0 {
                 warn!("Incomplete UTF-8 sequence interrupted by escape sequence byte 0x{:02X}", byte);
                 screen::handle_printable(self, REPLACEMENT_CHARACTER);
                 self.utf8_decoder.reset();
             }
            self.process_byte_in_parser(byte);
            if self.parser_state != initial_state {
                trace!(" -> State transition: {:?} -> {:?}", initial_state, self.parser_state);
            }
            return;
        }

        // --- Ground State Processing using Utf8Decoder ---
        match self.utf8_decoder.decode(byte) {
            Ok(Some(c)) => {
                // Successfully decoded a printable character
                screen::handle_printable(self, c);
            }
            Ok(None) => {
                // Incomplete sequence OR a C0/DEL control code identified by decoder
                if self.utf8_decoder.len == 0 {
                    // Decoder consumed the byte and signalled it's a control code (C0/DEL)
                    parser::handle_ground_control_or_esc(self, byte);
                } else {
                    // Incomplete sequence, just wait for more bytes
                    trace!("Incomplete UTF-8 sequence, waiting...");
                }
            }
            Err(()) => {
                // Invalid UTF-8 sequence detected by the decoder
                warn!("Invalid UTF-8 sequence detected at byte 0x{:02X}", byte);
                // Reset parser state (should be Ground, but safety first)
                self.parser_state = ParserState::Ground;
                parser::clear_csi_params(self);
                parser::clear_osc_string(self);
                // Handle the replacement character
                screen::handle_printable(self, REPLACEMENT_CHARACTER);
                // Decoder is already reset internally
            }
        }
         // Log state transition if it happened (e.g., Ground -> Escape)
         if self.parser_state != initial_state {
             trace!(" -> State transition: {:?} -> {:?}", initial_state, self.parser_state);
        }
    }


    /// Helper to pass a byte to the parser state machine (used internally).
    /// Called when the parser is not in the Ground state.
    fn process_byte_in_parser(&mut self, byte: u8) {
         match self.parser_state {
            ParserState::Ground => { warn!("process_byte_in_parser called in Ground state"); }
            ParserState::Escape => parser::handle_escape_byte(self, byte),
            ParserState::CSIEntry => parser::handle_csi_entry_byte(self, byte),
            ParserState::CSIParam => parser::handle_csi_param_byte(self, byte),
            ParserState::CSIIntermediate => parser::handle_csi_intermediate_byte(self, byte),
            ParserState::CSIIgnore => parser::handle_csi_ignore_byte(self, byte),
            ParserState::OSCString => parser::handle_osc_string_byte(self, byte),
        }
    }


     /// Resizes the terminal emulator state.
     pub fn resize(&mut self, new_width: usize, new_height: usize) {
         let old_height = self.height;
         let old_width = self.width;
         let new_height = max(1, new_height);
         let new_width = max(1, new_width);

         // Reset scroll region *before* calling screen::resize
         self.scroll_top = 0;
         self.scroll_bot = new_height.saturating_sub(1);

         if old_height == new_height && old_width == new_width {
             debug!("Resize called with same dimensions ({}, {}), only reset scroll region.", new_width, new_height);
             return;
         }
         debug!("Resizing terminal from {}x{} to {}x{}", old_width, old_height, new_width, new_height);

         screen::resize(self, new_width, new_height);
     }
}

// Add the module declaration for the tests file
#[cfg(test)]
mod tests;
