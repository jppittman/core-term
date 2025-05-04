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
const DEFAULT_TAB_INTERVAL: usize = 8;
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
    // Note: Cursor state like origin mode is now tracked in DecModes
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
    OSCParse,
    OSCParam,
    OSCString,
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
        Self::default() // Use Default trait
    }

    /// Resets the decoder state.
    fn reset(&mut self) {
        // No need for trace log here, called frequently
        self.len = 0;
    }

    /// Decodes the next byte in a potential UTF-8 sequence using `std::str::from_utf8`.
    ///
    /// Returns:
    /// * `Ok(Some(char))` if a character is successfully decoded.
    /// * `Ok(None)` if more bytes are needed for the current sequence.
    /// * `Err(())` if an invalid UTF-8 sequence is detected.
    fn decode(&mut self, byte: u8) -> Result<Option<char>, ()> {
        // trace!("Utf8Decoder: decode(0x{:02X}), current len={}", byte, self.len); // Too verbose

        // If the buffer is empty and we get ASCII, handle it directly
        if self.len == 0 && byte.is_ascii() {
            // trace!("Utf8Decoder: Direct ASCII '0x{:02X}' -> '{}'", byte, byte as char); // Too verbose
            return Ok(Some(byte as char));
        }

        // Prevent buffer overflow
        if self.len >= UTF8_BUFFER_SIZE {
            // trace!("Utf8Decoder: Buffer overflow (len={}), resetting and returning Err", self.len); // Too verbose
            self.reset();
            // Return error because we received a byte when buffer was full.
            return Err(());
        }

        // Add byte to buffer
        self.buffer[self.len] = byte;
        self.len += 1;
        // trace!("Utf8Decoder: Buffer = {:02X?}, len = {}", &self.buffer[0..self.len], self.len); // Too verbose


        // Try decoding the current buffer
        match str::from_utf8(&self.buffer[0..self.len]) {
            Ok(s) => {
                // Successfully decoded a valid UTF-8 sequence.
                let chr = s.chars().next(); // Should always succeed and contain exactly one char
                // trace!("Utf8Decoder: Decoded Ok('{}'), resetting.", chr.unwrap_or('?')); // Too verbose
                self.reset(); // Clear buffer as we consumed the character
                Ok(chr)
            }
            Err(e) => {
                // trace!("Utf8Decoder: from_utf8 returned Err: {:?}", e); // Too verbose
                if e.error_len().is_none() {
                    // trace!("Utf8Decoder: Incomplete sequence (error_len=None), returning Ok(None)."); // Too verbose
                    Ok(None)
                } else {
                    // trace!("Utf8Decoder: Invalid sequence (error_len=Some), resetting and returning Err."); // Too verbose
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
            origin_mode: false, // DECOM defaults to off (absolute mode)
        }
    }
}


// --- Main Term Struct Definition ---
#[allow(dead_code)] // Allow unused fields for now
pub struct Term {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) cursor: Cursor,
    pub(super) screen: Vec<Vec<Glyph>>,
    pub(super) alt_screen: Vec<Vec<Glyph>>,
    pub(super) using_alt_screen: bool,
    // Saved cursor/attributes state (for DECSC/DECRC, Alt Screen switching)
    pub(super) saved_cursor: Cursor,
    pub(super) saved_attributes: Attributes, // WARNING: unused field
    pub(super) saved_cursor_alt: Cursor,
    pub(super) saved_attributes_alt: Attributes, // WARNING: unused field
    // Current state
    pub(super) current_attributes: Attributes,
    pub(super) default_attributes: Attributes,
    // Parser state
    pub(super) parser_state: ParserState,
    pub(super) csi_params: Vec<u16>,
    pub(super) csi_intermediates: Vec<char>,
    pub(super) osc_string: String,
    // Use the Utf8Decoder struct again
    pub(super) utf8_decoder: Utf8Decoder,
    // Modes
    pub(super) dec_modes: DecModes,
    pub(super) tabs: Vec<bool>,
    // Scrolling region boundaries (inclusive, 0-based)
    pub(super) top: usize, // WARNING: unused field
    pub(super) bot: usize, // WARNING: unused field
    pub(super) scroll_top: usize,
    pub(super) scroll_bot: usize,

    // Dirty flags for rendering optimization
    pub(super) dirty: Vec<u8>, // WARNING: unused field
}

// --- Public API Implementation ---
impl Term {
    /// Creates a new terminal state with given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        let default_attributes = Attributes::default();
        let default_glyph = Glyph { c: ' ', attr: default_attributes };
        // Ensure minimum dimensions
        let width = max(1, width);
        let height = max(1, height);

        let screen = vec![vec![default_glyph; width]; height];
        let alt_screen = vec![vec![default_glyph; width]; height];
        let tabs = (0..width).map(|i| i % DEFAULT_TAB_INTERVAL == 0).collect();
        let dirty = vec![1u8; height]; // Mark all lines dirty initially

        Term {
            width, height, screen, alt_screen, using_alt_screen: false,
            cursor: Cursor::default(),
            // Initialize saved states
            saved_cursor: Cursor::default(),
            saved_attributes: default_attributes,
            saved_cursor_alt: Cursor::default(),
            saved_attributes_alt: default_attributes,
            // Initialize current state
            current_attributes: default_attributes,
            default_attributes,
            // Initialize parser state
            parser_state: ParserState::default(),
            csi_params: Vec::with_capacity(MAX_CSI_PARAMS),
            csi_intermediates: Vec::with_capacity(MAX_CSI_INTERMEDIATES),
            osc_string: String::with_capacity(MAX_OSC_STRING_LEN),
            // Initialize the Utf8Decoder
            utf8_decoder: Utf8Decoder::new(),
            // Initialize modes
            dec_modes: DecModes::default(), // Includes origin_mode = false
            tabs,
            // Initialize scrolling region to full screen
            top: 0,
            bot: height.saturating_sub(1), // Use saturating_sub for safety if height is 0
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
        // Log the whole chunk at DEBUG level
        if log::log_enabled!(log::Level::Debug) {
            let printable_bytes: String = bytes.iter().map(|&b| {
                if b.is_ascii_graphic() || b == b' ' {
                    b as char
                } else if b == 0x1b {
                    '␛' // ESC symbol
                } else if b < 0x20 || b == 0x7f {
                    '.' // Replacement for other C0/DEL
                } else {
                    '?' // Placeholder for non-ASCII or non-printable
                }
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
        // Reduce verbosity of per-byte logging, maybe use trace! if needed later
        // trace!("process_byte(0x{:02X} '{}'), state: {:?}", byte, if byte.is_ascii_graphic() || byte == b' ' { byte as char } else { '.' } , initial_state);

        match self.utf8_decoder.decode(byte) {
            Ok(Some(c)) => {
                // trace!(" -> Decoded char: '{}' (U+{:X})", c, c as u32); // Too verbose
                self.handle_decoded_char(c, byte);
            }
            Ok(None) => {
                // trace!(" -> Incomplete UTF-8 sequence, waiting for more bytes."); // Too verbose
            }
            Err(()) => {
                debug!(" -> Invalid UTF-8 sequence detected starting with 0x{:02X}!", byte);
                screen::handle_printable(self, REPLACEMENT_CHARACTER);
                if byte.is_ascii() {
                    // trace!(" -> Re-processing invalid byte 0x{:02X} as ASCII", byte); // Too verbose
                    self.handle_decoded_char(byte as char, byte);
                } else {
                     // trace!(" -> Discarding invalid non-ASCII byte 0x{:02X}", byte); // Too verbose
                }
            }
        }
        if self.parser_state != initial_state {
             trace!(" -> State transition: {:?} -> {:?}", initial_state, self.parser_state); // Keep state transitions at trace
        }
    }

    /// Handles a successfully decoded character.
    fn handle_decoded_char(&mut self, c: char, initiating_byte: u8) {
        match self.parser_state {
            ParserState::Ground => {
                if c == '\x1B' || // ESC
                   (c >= '\u{0080}' && c <= '\u{009F}') || // C1 controls
                   c < '\u{0020}' || c == '\u{007F}' // C0 controls
                {
                    parser::handle_osc_string_byte(self, initiating_byte);
                } else {
                    // Log printable chars at debug level
                    debug!(" -> Handling as printable character: '{}'", c);
                    screen::handle_printable(self, c);
                }
            }
            _ => {
                // trace!(" -> Passing byte 0x{:02X} to parser state {:?}", initiating_byte, self.parser_state); // Too verbose
                self.process_byte_in_parser(initiating_byte);
            }
        }
    }


    /// Helper to pass a byte to the parser state machine (used internally)
    fn process_byte_in_parser(&mut self, byte: u8) {
         match self.parser_state {
            ParserState::Ground => { warn!("process_byte_in_parser called in Ground state"); } , // Should not happen
            ParserState::Escape => parser::handle_escape_byte(self, byte),
            ParserState::CSIEntry => parser::handle_csi_entry_byte(self, byte),
            ParserState::CSIParam => parser::handle_csi_param_byte(self, byte),
            ParserState::CSIIntermediate => parser::handle_csi_intermediate_byte(self, byte),
            ParserState::CSIIgnore => parser::handle_csi_ignore_byte(self, byte),
            ParserState::OSCParam => parser::handle_osc_string_byte(self, byte),
            ParserState::OSCParse => parser::handle_osc_string_byte(self, byte),
            ParserState::OSCString => parser::handle_osc_string_byte(self, byte),
        }
    }


     /// Resizes the terminal emulator state.
     pub fn resize(&mut self, new_width: usize, new_height: usize) {
         screen::resize(self, new_width, new_height);
     }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    // Remove unused import: use test_log::test;

    #[test_log::test] // Use test_log attribute
    fn test_term_initialization() {
        let width = 80;
        let height = 24;
        let term = Term::new(width, height);
        assert_eq!(term.width, width);
        assert_eq!(term.height, height);
        assert_eq!(term.top, 0);
        assert_eq!(term.bot, height - 1);
        assert!(!term.dec_modes.origin_mode);
        assert!(!term.dec_modes.cursor_keys_app_mode);
        assert_eq!(term.cursor, Cursor { x: 0, y: 0 });
        assert_eq!(term.current_attributes, term.default_attributes);
        assert_eq!(term.saved_cursor, Cursor { x: 0, y: 0 });
        assert_eq!(term.saved_attributes, term.default_attributes);
        assert_eq!(term.saved_cursor_alt, Cursor { x: 0, y: 0 });
        assert_eq!(term.saved_attributes_alt, term.default_attributes);
        assert_eq!(term.screen.len(), height);
        assert_eq!(term.screen[0].len(), width);
        assert_eq!(term.alt_screen.len(), height);
        assert_eq!(term.alt_screen[0].len(), width);
        assert_eq!(term.dirty.len(), height);
        assert!(term.dirty.iter().all(|&d| d == 1));
        assert_eq!(term.tabs.len(), width);
        assert!(term.tabs[0]);
        assert!(term.tabs[DEFAULT_TAB_INTERVAL]);
        assert!(!term.tabs[1]);
        assert_eq!(term.utf8_decoder.len, 0);
    }

    #[test_log::test]
    fn test_utf8_decoder_ascii() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.decode(b'A'), Ok(Some('A')));
        assert_eq!(decoder.len, 0);
    }

     #[test_log::test]
    fn test_utf8_decoder_multi_byte_complete() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.decode(0xC3), Ok(None));
        assert_eq!(decoder.len, 1);
        assert_eq!(decoder.decode(0xA9), Ok(Some('é')));
        assert_eq!(decoder.len, 0);
    }

    #[test_log::test]
    fn test_utf8_decoder_multi_byte_incomplete() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.decode(0xE2), Ok(None));
        assert_eq!(decoder.len, 1);
        assert_eq!(decoder.decode(0x82), Ok(None));
        assert_eq!(decoder.len, 2);
        assert_eq!(decoder.decode(0xAC), Ok(Some('€')));
        assert_eq!(decoder.len, 0);
    }

    #[test_log::test]
    fn test_utf8_decoder_invalid_sequence_start() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.decode(0x80), Err(()));
        assert_eq!(decoder.len, 0);
        assert_eq!(decoder.decode(0xFF), Err(()));
        assert_eq!(decoder.len, 0);
    }

    #[test_log::test]
    fn test_utf8_decoder_invalid_sequence_mid() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.decode(0xC3), Ok(None));
        assert_eq!(decoder.decode(b'A'), Err(()));
        assert_eq!(decoder.len, 0);
        assert_eq!(decoder.decode(0xE2), Ok(None));
        assert_eq!(decoder.decode(0xC3), Err(()));
        assert_eq!(decoder.len, 0);
    }

    #[test_log::test]
    fn test_utf8_decoder_max_bytes() {
         let mut decoder = Utf8Decoder::new();
         let label_bytes = [0xF0, 0x9F, 0x8F, 0xB7];
         assert_eq!(decoder.decode(label_bytes[0]), Ok(None));
         assert_eq!(decoder.decode(label_bytes[1]), Ok(None));
         assert_eq!(decoder.decode(label_bytes[2]), Ok(None));
         assert_eq!(decoder.decode(label_bytes[3]), Ok(Some('🏷')));
         assert_eq!(decoder.len, 0);
         assert_eq!(decoder.decode(0xF0), Ok(None));
         assert_eq!(decoder.decode(0x9F), Ok(None));
         assert_eq!(decoder.decode(0x8F), Ok(None));
         assert_eq!(decoder.decode(0xB7), Ok(Some('🏷')));
         assert_eq!(decoder.len, 0);
         assert_eq!(decoder.decode(0x01), Ok(Some('\u{1}')));
         assert_eq!(decoder.len, 0);
    }

    #[test_log::test]
    fn test_utf8_decoder_reset_on_error() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.decode(0xE2), Ok(None));
        assert_eq!(decoder.len, 1);
        assert_eq!(decoder.decode(0x41), Err(()));
        assert_eq!(decoder.len, 0);
        assert_eq!(decoder.decode(b'B'), Ok(Some('B')));
        assert_eq!(decoder.len, 0);
    }

    #[test_log::test]
    fn test_process_byte_ascii_printable_via_term() {
        let mut term = Term::new(10, 1);
        term.process_byte(b'A');
        term.process_byte(b'B');
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'A');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, 'B');
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.utf8_decoder.len, 0);
    }

     #[test_log::test]
    fn test_process_byte_ascii_control_via_term() {
        let mut term = Term::new(10, 2);
        term.process_byte(b'A');
        term.process_byte(b'\n');
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'A');
        assert_eq!(term.cursor.x, 0);
        assert_eq!(term.cursor.y, 1);
        assert_eq!(term.utf8_decoder.len, 0);
    }

    #[test_log::test]
    fn test_process_byte_multi_byte_complete_via_term() {
        let mut term = Term::new(10, 1);
        term.process_byte(0xC3);
        assert_eq!(term.utf8_decoder.len, 1);
        assert_eq!(term.cursor.x, 0);
        term.process_byte(0xA9);
        assert_eq!(term.utf8_decoder.len, 0);
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'é');
        assert_eq!(term.cursor.x, 1);
    }

    #[test_log::test]
    fn test_process_byte_invalid_sequence_via_term() {
        let mut term = Term::new(10, 1);
        term.process_byte(0xC3);
        assert_eq!(term.utf8_decoder.len, 1);
        term.process_byte(0x20);
        assert_eq!(term.utf8_decoder.len, 0);
        assert_eq!(term.get_glyph(0, 0).unwrap().c, REPLACEMENT_CHARACTER);
        assert_eq!(term.get_glyph(1, 0).unwrap().c, ' ');
        assert_eq!(term.cursor.x, 2);
    }

     #[test_log::test]
    fn test_process_byte_invalid_start_byte_via_term() {
        let mut term = Term::new(10, 1);
        term.process_byte(0x80);
        assert_eq!(term.utf8_decoder.len, 0);
        assert_eq!(term.get_glyph(0, 0).unwrap().c, REPLACEMENT_CHARACTER);
        assert_eq!(term.cursor.x, 1);
        term.process_byte(b'X');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, 'X');
        assert_eq!(term.cursor.x, 2);

    }
}
