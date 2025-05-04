// src/term/mod.rs

//! Defines the core `Term` struct, representing the terminal emulator's state,
//! and handles the overall processing of input bytes through its state machine.

// Declare submodules (implementations will go in separate files)
mod parser;
mod screen;

// Use necessary items from glyph module
use crate::glyph::{Glyph, Attributes, REPLACEMENT_CHARACTER};
use std::str;
use std::cmp::max;
use log::{trace, debug, warn};

// --- Constants ---
/// Default tab stop interval.
pub(super) const DEFAULT_TAB_INTERVAL: usize = 8;
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
}

/// Holds DEC Private Mode settings.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(super) struct DecModes {
    /// DECCKM state (false = normal, true = application cursor keys)
    cursor_keys_app_mode: bool,
    /// DECOM state (false = absolute coords, true = relative to scroll region)
    origin_mode: bool,
}

// --- Utf8Decoder Implementation ---
impl Utf8Decoder {
    /// Creates a new, empty UTF-8 decoder.
    fn new() -> Self {
        Self::default()
    }

    /// Resets the decoder state, clearing any buffered bytes.
    fn reset(&mut self) {
        self.len = 0;
    }

    /// Decodes the next byte in a potential UTF-8 sequence.
    ///
    /// # Returns
    /// * `Ok(Some(char))` - If a complete, valid UTF-8 character was decoded.
    /// * `Ok(None)` - If the byte is part of an incomplete sequence OR if it's a C0/DEL control code.
    /// * `Err(())` - If the byte results in an invalid UTF-8 sequence.
    fn decode(&mut self, byte: u8) -> Result<Option<char>, ()> {
        // Handle simple ASCII case first (optimization and handles control codes)
        if self.len == 0 && byte < 0x80 {
            // C0 controls (0x00-0x1F) and DEL (0x7F) are returned as Ok(None)
            // to signal they should be handled by the control code logic.
            if byte < 0x20 || byte == parser::DEL {
                return Ok(None);
            } else {
                // Printable ASCII (0x20-0x7E)
                return Ok(Some(byte as char));
            }
        }

        // Check for buffer overflow before adding the new byte
        if self.len >= UTF8_BUFFER_SIZE {
            warn!("UTF-8 decoder buffer overflow, resetting.");
            self.reset();
            // Retry decoding the current byte now that the buffer is clear.
            // This handles cases where a valid sequence start byte arrives
            // immediately after a sequence that filled the buffer.
            return self.decode(byte);
        }

        self.buffer[self.len] = byte;
        self.len += 1;

        match str::from_utf8(&self.buffer[0..self.len]) {
            Ok(s) => {
                let c = s.chars().next(); // Should always contain exactly one char
                self.reset();
                Ok(c)
            }
            Err(e) => {
                // `e.error_len().is_none()` means the error is potentially recoverable
                // if more bytes arrive (i.e., it's an incomplete sequence).
                if e.error_len().is_none() && self.len < UTF8_BUFFER_SIZE {
                    // Valid start, but needs more bytes and buffer is not full.
                    Ok(None)
                } else {
                    // Invalid sequence (invalid byte, sequence too long, overlong encoding, etc.)
                    // The caller should handle this error.
                    self.reset();
                    Err(())
                }
            }
        }
    }
}

// --- Main Term Struct Definition ---
/// Represents the state of the terminal emulator.
pub struct Term {
    /// Terminal width in columns.
    pub(super) width: usize,
    /// Terminal height in rows.
    pub(super) height: usize,
    /// Flag indicating if the cursor is currently wrapped to the next line logically.
    /// Set when a character is written in the last column. Reset by explicit cursor movement.
    pub(super) wrap_next: bool,
    /// Current cursor position.
    pub(super) cursor: Cursor,
    /// Primary screen buffer.
    pub(super) screen: Vec<Vec<Glyph>>,
    /// Alternate screen buffer.
    pub(super) alt_screen: Vec<Vec<Glyph>>,
    /// Flag indicating if the alternate screen buffer is currently active.
    pub(super) using_alt_screen: bool,
    /// Saved cursor position for the primary screen (used by DECSC/DECRC, SCORC/SCOSC).
    pub(super) saved_cursor: Cursor,
    /// Saved attributes for the primary screen.
    #[allow(dead_code)] // Potentially useful later, silence warning for now
    pub(super) saved_attributes: Attributes,
    /// Saved cursor position for the alternate screen.
    pub(super) saved_cursor_alt: Cursor,
    /// Saved attributes for the alternate screen.
    #[allow(dead_code)] // Potentially useful later, silence warning for now
    pub(super) saved_attributes_alt: Attributes,
    /// Current character attributes (bold, colors, etc.) to apply to new glyphs.
    pub(super) current_attributes: Attributes,
    /// Default character attributes (used for clearing, reset).
    pub(super) default_attributes: Attributes,
    /// Current state of the escape sequence parser.
    pub(super) parser_state: ParserState,
    /// Collected parameters for the current CSI sequence.
    pub(super) csi_params: Vec<u16>,
    /// Collected intermediate bytes for the current CSI sequence.
    pub(super) csi_intermediates: Vec<char>,
    /// Collected string for the current OSC sequence.
    pub(super) osc_string: String,
    /// UTF-8 decoder state.
    pub(super) utf8_decoder: Utf8Decoder,
    /// DEC private mode settings.
    pub(super) dec_modes: DecModes,
    /// Tab stop positions (true if a tab stop exists at the index).
    pub(super) tabs: Vec<bool>,
    /// Top line of the scrolling region (0-based, inclusive).
    pub(super) scroll_top: usize,
    /// Bottom line of the scrolling region (0-based, inclusive).
    pub(super) scroll_bot: usize,
    /// Tracks which lines need redrawing (implementation detail, may change).
    #[allow(dead_code)] // Potentially useful later, silence warning for now
    pub(super) dirty: Vec<u8>,
}

// --- Public API Implementation ---
impl Term {
    /// Creates a new terminal state with given dimensions.
    /// Minimum dimensions are 1x1.
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
            wrap_next: false,
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
    /// Returns `None` if coordinates are out of bounds.
    pub fn get_glyph(&self, x: usize, y: usize) -> Option<&Glyph> {
        let current_screen = if self.using_alt_screen { &self.alt_screen } else { &self.screen };
        current_screen.get(y).and_then(|row| row.get(x))
    }

    /// Gets the current cursor position (0-based column, 0-based row).
    pub fn get_cursor(&self) -> (usize, usize) { (self.cursor.x, self.cursor.y) }

    /// Gets the terminal dimensions (width, height) in cells.
    pub fn get_dimensions(&self) -> (usize, usize) { (self.width, self.height) }

    /// Processes a slice of bytes received from the PTY or input source.
    /// Iterates through the bytes, calling `process_byte` for each.
    pub fn process_bytes(&mut self, bytes: &[u8]) {
        if log::log_enabled!(log::Level::Debug) {
            // Log a sanitized version for easier debugging
            let printable_bytes: String = bytes.iter().map(|&b| {
                if b.is_ascii_graphic() || b == b' ' { b as char }
                else if b == parser::ESC { '␛' } // Represent ESC clearly
                else if b < 0x20 || b == parser::DEL { '.' } // Represent C0/DEL as '.'
                else { '?' } // Represent other non-printable as '?'
            }).collect();
            debug!("Processing {} bytes from PTY: '{}'", bytes.len(), printable_bytes);
        }
        for &byte in bytes {
            self.process_byte(byte);
        }
    }

    /// Processes a single byte received from the PTY or input source.
    /// Handles UTF-8 decoding and state machine transitions.
    pub fn process_byte(&mut self, byte: u8) {
        let initial_state = self.parser_state;
        trace!("process_byte: byte=0x{:02X}, state={:?}", byte, initial_state);

        // If not in Ground state, pass byte directly to the parser state machine handlers.
        // Check for and handle incomplete UTF-8 sequences interrupted by escape codes.
        if initial_state != ParserState::Ground {
             if self.utf8_decoder.len > 0 {
                 // An escape sequence byte arrived while decoding a multi-byte char.
                 warn!("Incomplete UTF-8 sequence interrupted by escape sequence byte 0x{:02X}", byte);
                 // Print a replacement character for the incomplete sequence.
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
                // Successfully decoded a printable character.
                screen::handle_printable(self, c);
            }
            Ok(None) => {
                // Incomplete sequence OR a C0/DEL control code identified by decoder.
                if self.utf8_decoder.len == 0 {
                    // Decoder consumed the byte and signalled it's a control code (C0/DEL) or ESC.
                    // Handle C0 controls or transition to Escape state.
                    parser::handle_ground_control_or_esc(self, byte);
                }
                // If utf8_decoder.len > 0, it was an incomplete sequence, just wait for more bytes.
            }
            Err(()) => {
                // Invalid UTF-8 sequence detected by the decoder.
                warn!("Invalid UTF-8 sequence detected at byte 0x{:02X}", byte);
                // Reset parser state if it somehow got corrupted (should be Ground).
                self.parser_state = ParserState::Ground;
                parser::clear_csi_params(self);
                parser::clear_osc_string(self);
                // Handle the replacement character.
                screen::handle_printable(self, REPLACEMENT_CHARACTER);
                // Decoder is already reset internally by the Err return.
            }
        }
         // Log state transition if it happened (e.g., Ground -> Escape)
         if self.parser_state != initial_state {
             trace!(" -> State transition: {:?} -> {:?}", initial_state, self.parser_state);
        }
    }


    /// Helper to pass a byte to the parser state machine handlers (used internally).
    /// This is called only when the parser is *not* in the Ground state.
    fn process_byte_in_parser(&mut self, byte: u8) {
         match self.parser_state {
            // This function should not be called in Ground state.
            ParserState::Ground => { warn!("process_byte_in_parser called unexpectedly in Ground state"); }
            ParserState::Escape => parser::handle_escape_byte(self, byte),
            ParserState::CSIEntry => parser::handle_csi_entry_byte(self, byte),
            ParserState::CSIParam => parser::handle_csi_param_byte(self, byte),
            ParserState::CSIIntermediate => parser::handle_csi_intermediate_byte(self, byte),
            ParserState::CSIIgnore => parser::handle_csi_ignore_byte(self, byte),
            ParserState::OSCString => parser::handle_osc_string_byte(self, byte),
        }
    }


     /// Resizes the terminal emulator state, including screen buffers and cursor position.
     /// Resets the scrolling region to the full screen size.
     pub fn resize(&mut self, new_width: usize, new_height: usize) {
         let old_height = self.height;
         let old_width = self.width;
         // Ensure minimum dimensions of 1x1.
         let new_height = max(1, new_height);
         let new_width = max(1, new_width);

         // Reset scroll region *before* calling screen::resize, as screen::resize
         // might clamp cursor positions based on the old region if called first.
         self.scroll_top = 0;
         self.scroll_bot = new_height.saturating_sub(1);

         if old_height == new_height && old_width == new_width {
             debug!("Resize called with same dimensions ({}, {}), only reset scroll region.", new_width, new_height);
             return;
         }
         debug!("Resizing terminal from {}x{} to {}x{}", old_width, old_height, new_width, new_height);

         // Call the screen submodule's resize logic to handle buffer adjustments.
         screen::resize(self, new_width, new_height);

         // Update the main Term struct's dimensions *after* screen::resize.
         self.width = new_width;
         self.height = new_height;
     }
}

// Add the module declaration for the tests file
#[cfg(test)]
mod tests;
