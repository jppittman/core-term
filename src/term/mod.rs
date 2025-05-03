// src/term/mod.rs

// Declare submodules (implementations will go in separate files)
mod parser;
mod screen;

// Use necessary items from glyph module
use crate::glyph::{Glyph, Attributes, REPLACEMENT_CHARACTER};
use std::str;
use std::cmp::max; // Import max directly
// Add log crate import
use log::trace;

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
        trace!("Utf8Decoder: Resetting state (len was {})", self.len);
        self.len = 0;
        // No need to clear the buffer contents explicitly
    }

    /// Decodes the next byte in a potential UTF-8 sequence using `std::str::from_utf8`.
    ///
    /// Returns:
    /// * `Ok(Some(char))` if a character is successfully decoded.
    /// * `Ok(None)` if more bytes are needed for the current sequence.
    /// * `Err(())` if an invalid UTF-8 sequence is detected.
    fn decode(&mut self, byte: u8) -> Result<Option<char>, ()> {
        trace!("Utf8Decoder: decode(0x{:02X}), current len={}", byte, self.len);

        // If the buffer is empty and we get ASCII, handle it directly
        if self.len == 0 && byte.is_ascii() {
            trace!("Utf8Decoder: Direct ASCII '0x{:02X}' -> '{}'", byte, byte as char);
            return Ok(Some(byte as char));
        }

        // Prevent buffer overflow
        if self.len >= UTF8_BUFFER_SIZE {
            trace!("Utf8Decoder: Buffer overflow (len={}), resetting and returning Err", self.len);
            self.reset();
            // Return error because we received a byte when buffer was full.
            return Err(());
        }

        // Add byte to buffer
        self.buffer[self.len] = byte;
        self.len += 1;
        trace!("Utf8Decoder: Buffer = {:02X?}, len = {}", &self.buffer[0..self.len], self.len);


        // Try decoding the current buffer
        match str::from_utf8(&self.buffer[0..self.len]) {
            Ok(s) => {
                // Successfully decoded a valid UTF-8 sequence.
                // Since from_utf8 succeeded, this *must* be a complete character.
                let chr = s.chars().next(); // Should always succeed and contain exactly one char
                trace!("Utf8Decoder: Decoded Ok('{}'), resetting.", chr.unwrap_or('?'));
                self.reset(); // Clear buffer as we consumed the character
                Ok(chr)
            }
            Err(e) => {
                trace!("Utf8Decoder: from_utf8 returned Err: {:?}", e);
                // from_utf8 failed. Check if it's because we simply need more bytes,
                // or if the sequence is truly invalid.
                // error_len() is None means "valid UTF-8 prefix" -> incomplete.
                if e.error_len().is_none() {
                    trace!("Utf8Decoder: Incomplete sequence (error_len=None), returning Ok(None).");
                    // Incomplete sequence error according to from_utf8.
                    // Stay in the current state and wait for more bytes.
                    Ok(None)
                } else {
                    trace!("Utf8Decoder: Invalid sequence (error_len=Some), resetting and returning Err.");
                    // Invalid sequence detected by from_utf8 (e.g., invalid byte, overlong).
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
pub struct Term {
    // Make fields pub(super) so screen.rs and parser.rs can access them
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) cursor: Cursor,
    pub(super) screen: Vec<Vec<Glyph>>,
    pub(super) alt_screen: Vec<Vec<Glyph>>,
    pub(super) using_alt_screen: bool,
    // Saved cursor/attributes state (for DECSC/DECRC, Alt Screen switching)
    pub(super) saved_cursor: Cursor,
    pub(super) saved_attributes: Attributes,
    pub(super) saved_cursor_alt: Cursor,
    pub(super) saved_attributes_alt: Attributes,
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
    pub(super) top: usize,
    pub(super) bot: usize,
    // Dirty flags for rendering optimization
    pub(super) dirty: Vec<u8>, // Using u8 for simplicity (0=clean, 1=dirty)
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
        for &byte in bytes {
            self.process_byte(byte);
        }
    }

    /// Processes a single byte received from the PTY or input source.
    /// Handles UTF-8 decoding using the Utf8Decoder and delegates
    /// to escape sequence parser or character handler.
    pub fn process_byte(&mut self, byte: u8) {
        // Always feed the byte to the decoder first
        match self.utf8_decoder.decode(byte) {
            Ok(Some(c)) => {
                // Successfully decoded a character
                self.handle_decoded_char(c, byte);
            }
            Ok(None) => {
                // Incomplete sequence, wait for more bytes. Do nothing.
            }
            Err(()) => {
                // Invalid sequence detected by the decoder.
                // Handle the error by printing a replacement character.
                screen::handle_printable(self, REPLACEMENT_CHARACTER);
                // The decoder reset itself internally.
                // Now, try to process the byte that *caused* the error
                // as if it were a new character, but only if it's valid standalone.
                if byte.is_ascii() {
                    // Re-process the ASCII byte that broke the sequence
                    self.handle_decoded_char(byte as char, byte);
                }
                // If the error-causing byte was not ASCII, it's discarded.
            }
        }
    }

    /// Handles a successfully decoded character.
    /// Decides whether to pass to parser state machine or handle as printable.
    /// `initiating_byte` is the first byte of the sequence for this char,
    /// needed for passing to parser handlers like handle_ground_byte.
    fn handle_decoded_char(&mut self, c: char, initiating_byte: u8) {
        match self.parser_state {
            ParserState::Ground => {
                // Check for C0/C1 controls or escape sequences
                if c == '\x1B' || // ESC
                   (c >= '\u{0080}' && c <= '\u{009F}') || // C1 controls
                   c < '\u{0020}' || c == '\u{007F}' // C0 controls
                {
                    // Pass the *byte* that initiated the sequence/control
                    parser::handle_ground_byte(self, initiating_byte);
                } else {
                    // Handle printable character
                    screen::handle_printable(self, c);
                }
            }
            // If already in a sequence, let the parser handle the raw byte
            // that completed the character.
            _ => {
                self.process_byte_in_parser(initiating_byte);
            }
        }
    }


    /// Helper to pass a byte to the parser state machine (used internally)
    fn process_byte_in_parser(&mut self, byte: u8) {
         match self.parser_state {
            // Ground state already handled in handle_decoded_char
            ParserState::Ground => { /* Should not be reached here */ } ,
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
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*; // Import items from parent module (mod.rs)
    // Remove unused import: use test_log::test;

    #[test_log::test] // Use test_log attribute
    fn test_term_initialization() {
        let width = 80;
        let height = 24;
        let term = Term::new(width, height);

        // Check dimensions
        assert_eq!(term.width, width);
        assert_eq!(term.height, height);

        // Check default scrolling region
        assert_eq!(term.top, 0, "Initial top margin should be 0");
        assert_eq!(term.bot, height - 1, "Initial bottom margin should be height - 1");

        // Check default DEC modes
        assert!(!term.dec_modes.origin_mode, "Origin mode (DECOM) should be off by default");
        assert!(!term.dec_modes.cursor_keys_app_mode, "Cursor keys app mode should be off by default");

        // Check default cursor position
        assert_eq!(term.cursor, Cursor { x: 0, y: 0 }, "Cursor should start at (0, 0)");

        // Check default attributes
        assert_eq!(term.current_attributes, term.default_attributes, "Current attributes should match default");

        // Check saved states initialization
        assert_eq!(term.saved_cursor, Cursor { x: 0, y: 0 }, "Saved cursor should be (0, 0)");
        assert_eq!(term.saved_attributes, term.default_attributes, "Saved attributes should match default");
        assert_eq!(term.saved_cursor_alt, Cursor { x: 0, y: 0 }, "Alt saved cursor should be (0, 0)");
        assert_eq!(term.saved_attributes_alt, term.default_attributes, "Alt saved attributes should match default");

        // Check screen buffer sizes
        assert_eq!(term.screen.len(), height, "Screen height mismatch");
        assert_eq!(term.screen[0].len(), width, "Screen width mismatch");
        assert_eq!(term.alt_screen.len(), height, "Alt screen height mismatch");
        assert_eq!(term.alt_screen[0].len(), width, "Alt screen width mismatch");

        // Check dirty flags
        assert_eq!(term.dirty.len(), height, "Dirty flags height mismatch");
        assert!(term.dirty.iter().all(|&d| d == 1), "All lines should be dirty initially");

        // Check tabs
        assert_eq!(term.tabs.len(), width);
        assert!(term.tabs[0]); // Tab stop at column 0
        assert!(term.tabs[DEFAULT_TAB_INTERVAL]); // Tab stop at default interval
        assert!(!term.tabs[1]); // No tab stop at column 1

        // Check Utf8Decoder state
        assert_eq!(term.utf8_decoder.len, 0);
    }

    // --- Keep the original Utf8Decoder tests, now testing the std wrapper ---
    #[test_log::test] // Use test_log attribute
    fn test_utf8_decoder_ascii() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.decode(b'A'), Ok(Some('A')));
        assert_eq!(decoder.len, 0);
    }

     #[test_log::test] // Use test_log attribute
    fn test_utf8_decoder_multi_byte_complete() {
        let mut decoder = Utf8Decoder::new();
        // Example: é (0xC3 0xA9)
        assert_eq!(decoder.decode(0xC3), Ok(None)); // Need more bytes
        assert_eq!(decoder.len, 1);
        assert_eq!(decoder.decode(0xA9), Ok(Some('é')));
        assert_eq!(decoder.len, 0); // Decoder reset
    }

    #[test_log::test] // Use test_log attribute
    fn test_utf8_decoder_multi_byte_incomplete() {
        let mut decoder = Utf8Decoder::new();
        // Example: Start of € (0xE2 0x82 0xAC)
        assert_eq!(decoder.decode(0xE2), Ok(None));
        assert_eq!(decoder.len, 1);
        assert_eq!(decoder.decode(0x82), Ok(None));
        assert_eq!(decoder.len, 2);
        // Still need 0xAC -> should return Ok(None)
        assert_eq!(decoder.decode(0xAC), Ok(Some('€'))); // Complete
        assert_eq!(decoder.len, 0);
    }

    #[test_log::test] // Use test_log attribute
    fn test_utf8_decoder_invalid_sequence_start() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.decode(0x80), Err(())); // Invalid start byte (continuation byte)
        assert_eq!(decoder.len, 0);
        assert_eq!(decoder.decode(0xFF), Err(())); // Invalid start byte
        assert_eq!(decoder.len, 0);
    }

    #[test_log::test] // Use test_log attribute
    fn test_utf8_decoder_invalid_sequence_mid() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.decode(0xC3), Ok(None)); // Expecting continuation byte
        assert_eq!(decoder.decode(b'A'), Err(())); // Invalid continuation byte (ASCII)
        assert_eq!(decoder.len, 0);

        assert_eq!(decoder.decode(0xE2), Ok(None)); // Expecting 2 continuation bytes
        assert_eq!(decoder.decode(0xC3), Err(())); // Invalid continuation byte (start byte)
        assert_eq!(decoder.len, 0);
    }

    #[test_log::test] // Use test_log attribute
    fn test_utf8_decoder_max_bytes() {
         let mut decoder = Utf8Decoder::new();
         // Valid 4-byte sequence for Label emoji 🏷 (U+1F3F7)
         let label_bytes = [0xF0, 0x9F, 0x8F, 0xB7];
         assert_eq!(decoder.decode(label_bytes[0]), Ok(None));
         assert_eq!(decoder.decode(label_bytes[1]), Ok(None));
         assert_eq!(decoder.decode(label_bytes[2]), Ok(None));
         // Correct test assertion: Expect Ok(Some('🏷')) for complete char
         assert_eq!(decoder.decode(label_bytes[3]), Ok(Some('🏷')));
         assert_eq!(decoder.len, 0);

         // Try feeding 5 bytes (invalid)
         // Re-feed the 4 bytes, then the 5th
         assert_eq!(decoder.decode(0xF0), Ok(None));
         assert_eq!(decoder.decode(0x9F), Ok(None));
         assert_eq!(decoder.decode(0x8F), Ok(None));
         assert_eq!(decoder.decode(0xB7), Ok(Some('🏷'))); // Should succeed here
         assert_eq!(decoder.len, 0); // Should reset
         // The 5th byte (0x01) is now processed as a new character
         assert_eq!(decoder.decode(0x01), Ok(Some('\u{1}')));
         assert_eq!(decoder.len, 0);
    }

    #[test_log::test] // Use test_log attribute
    fn test_utf8_decoder_reset_on_error() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.decode(0xE2), Ok(None)); // Start 3-byte
        assert_eq!(decoder.len, 1);
        assert_eq!(decoder.decode(0x41), Err(())); // Invalid continuation
        assert_eq!(decoder.len, 0); // Decoder should be reset
        assert_eq!(decoder.decode(b'B'), Ok(Some('B'))); // Should decode next byte correctly
        assert_eq!(decoder.len, 0);
    }

    // --- Tests for process_byte logic (using the new decoder) ---

    #[test_log::test] // Use test_log attribute
    fn test_process_byte_ascii_printable_via_term() {
        let mut term = Term::new(10, 1);
        term.process_byte(b'A');
        term.process_byte(b'B');
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'A');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, 'B');
        assert_eq!(term.cursor.x, 2);
        assert_eq!(term.utf8_decoder.len, 0);
    }

     #[test_log::test] // Use test_log attribute
    fn test_process_byte_ascii_control_via_term() {
        let mut term = Term::new(10, 2);
        term.process_byte(b'A');
        term.process_byte(b'\n'); // LF
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'A');
        assert_eq!(term.cursor.x, 0); // Moved by newline
        assert_eq!(term.cursor.y, 1); // Moved by newline
        assert_eq!(term.utf8_decoder.len, 0);
    }

    #[test_log::test] // Use test_log attribute
    fn test_process_byte_multi_byte_complete_via_term() {
        let mut term = Term::new(10, 1);
        term.process_byte(0xC3); // Start 'é'
        assert_eq!(term.utf8_decoder.len, 1);
        assert_eq!(term.cursor.x, 0); // No char processed yet
        term.process_byte(0xA9); // Complete 'é'
        assert_eq!(term.utf8_decoder.len, 0); // Decoder reset
        assert_eq!(term.get_glyph(0, 0).unwrap().c, 'é');
        assert_eq!(term.cursor.x, 1); // Cursor advanced
    }

    #[test_log::test] // Use test_log attribute
    fn test_process_byte_invalid_sequence_via_term() {
        let mut term = Term::new(10, 1);
        term.process_byte(0xC3); // Start 'é'
        assert_eq!(term.utf8_decoder.len, 1);
        term.process_byte(0x20); // Invalid continuation (space)
        assert_eq!(term.utf8_decoder.len, 0); // Decoder reset on error
        // Should have printed replacement char for the invalid sequence
        assert_eq!(term.get_glyph(0, 0).unwrap().c, REPLACEMENT_CHARACTER);
        // The space itself should then be processed because it's ASCII
        assert_eq!(term.get_glyph(1, 0).unwrap().c, ' ');
        // Correct test assertion: Cursor should be at x=2 after replacement + space
        assert_eq!(term.cursor.x, 2);
    }

     #[test_log::test] // Use test_log attribute
    fn test_process_byte_invalid_start_byte_via_term() {
        let mut term = Term::new(10, 1);
        term.process_byte(0x80); // Invalid start byte
        assert_eq!(term.utf8_decoder.len, 0); // Decoder reset on error
        assert_eq!(term.get_glyph(0, 0).unwrap().c, REPLACEMENT_CHARACTER);
        assert_eq!(term.cursor.x, 1);
        // Ensure subsequent valid byte works
        term.process_byte(b'X');
        assert_eq!(term.get_glyph(1, 0).unwrap().c, 'X');
        assert_eq!(term.cursor.x, 2);

    }
}

