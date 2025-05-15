// src/ansi/lexer.rs

//! ANSI escape sequence lexer.
//! Converts a byte stream into `AnsiToken`s, processing byte by byte,
//! handling UTF-8 decoding and state across calls.

use std::{mem, str};

/// Unicode replacement character (U+FFFD).
/// Used when encountering invalid UTF-8 sequences.
const REPLACEMENT_CHARACTER: char = '\u{FFFD}';

// --- Constants for Unicode boundaries ---
const UNICODE_MAX_CODE_POINT: u32 = 0x10FFFF;
const UNICODE_SURROGATE_START: u32 = 0xD800;
const UNICODE_SURROGATE_END: u32 = 0xDFFF;

// --- Constants for UTF-8 byte classification ---
const UTF8_ASCII_MAX: u8 = 0x7F;
const UTF8_CONT_MIN: u8 = 0x80;
const UTF8_CONT_MAX: u8 = 0xBF;
const UTF8_2_BYTE_MIN: u8 = 0xC2; // Excludes overlong 0xC0, 0xC1
const UTF8_3_BYTE_MIN: u8 = 0xE0;
const UTF8_4_BYTE_MIN: u8 = 0xF0;
const UTF8_INVALID_MIN: u8 = 0xF5; // Start of invalid range

/// Represents a single token identified by the lexer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnsiToken {
    /// A printable character, decoded from UTF-8.
    Print(char),
    /// A C0 control code (0x00 - 0x1F, plus DEL 0x7F).
    C0Control(u8),
    /// A C1 control code (0x80 - 0x9F).
    C1Control(u8),
}

/// Internal state machine for decoding UTF-8 byte streams incrementally.
#[derive(Debug, Clone, Default)]
struct Utf8Decoder {
    /// Buffer to hold bytes of a potentially multi-byte character.
    buffer: [u8; 4],
    /// Number of bytes currently stored in the buffer.
    len: usize,
    /// Total number of bytes expected for the current character sequence.
    expected: usize,
}

impl Utf8Decoder {
    /// Processes a single byte, attempting to decode a UTF-8 character.
    ///
    /// Maintains internal state across calls.
    ///
    /// # Returns
    ///
    /// * `Some(char)`: If a complete, valid Unicode character (excluding surrogates)
    ///   is decoded. The internal state is reset.
    /// * `Some(REPLACEMENT_CHARACTER)`: If an invalid byte sequence is detected
    ///   (invalid start byte, invalid continuation, overlong sequence, surrogate, etc.).
    ///   The internal state is reset.
    /// * `None`: If the byte is valid so far, but more bytes are needed to complete
    ///   the character. The internal state is updated.
    fn decode(&mut self, byte: u8) -> Option<char> {
        // If len is 0, we are processing the first byte of a potential sequence.
        if self.len == 0 {
            return self.decode_first_byte(byte);
        }

        // Otherwise, we are processing a continuation byte.
        self.decode_continuation_byte(byte)
    }

    /// Handles the first byte of a potential UTF-8 sequence.
    #[inline]
    fn decode_first_byte(&mut self, byte: u8) -> Option<char> {
        match byte {
            0..=UTF8_ASCII_MAX => {
                Some(byte as char) // ASCII is always valid and complete
            }
            UTF8_CONT_MIN..=0xC1 => {
                self.reset(); // Invalid start byte
                Some(REPLACEMENT_CHARACTER)
            }
            UTF8_2_BYTE_MIN..=0xDF => {
                self.expected = 2;
                self.buffer[0] = byte;
                self.len = 1;
                None // Need more bytes
            }
            UTF8_3_BYTE_MIN..=0xEF => {
                self.expected = 3;
                self.buffer[0] = byte;
                self.len = 1;
                None // Need more bytes
            }
            UTF8_4_BYTE_MIN..=0xF4 => {
                self.expected = 4;
                self.buffer[0] = byte;
                self.len = 1;
                None // Need more bytes
            }
            UTF8_INVALID_MIN..=0xFF => {
                self.reset(); // Invalid start byte
                Some(REPLACEMENT_CHARACTER)
            }
        }
    }

    /// Handles a continuation byte of a UTF-8 sequence.
    #[inline]
    fn decode_continuation_byte(&mut self, byte: u8) -> Option<char> {
        match byte {
            UTF8_CONT_MIN..=UTF8_CONT_MAX => {
                // Valid continuation byte
                self.buffer[self.len] = byte;
                self.len += 1;

                // Use guard clause: If sequence is still incomplete, return None early.
                if self.len != self.expected {
                    return None; // Need more bytes
                }

                // Sequence complete, attempt decode and validation
                let result = match str::from_utf8(&self.buffer[0..self.len]) {
                    Ok(s) => match s.chars().next() {
                        Some(c) => {
                            let cp = c as u32;
                            // Validate code point range and non-surrogate status
                            if cp <= UNICODE_MAX_CODE_POINT
                                && !(UNICODE_SURROGATE_START..=UNICODE_SURROGATE_END).contains(&cp)
                            {
                                Some(c)
                            } else {
                                Some(REPLACEMENT_CHARACTER) // Invalid code point
                            }
                        }
                        None => Some(REPLACEMENT_CHARACTER), // Should not happen
                    },
                    Err(_) => Some(REPLACEMENT_CHARACTER), // Invalid UTF-8 sequence bytes
                };
                self.reset(); // Reset state after completion or error
                result // Return the decoded char or replacement
            }
            _ => {
                // Invalid continuation byte
                self.reset();
                Some(REPLACEMENT_CHARACTER)
            }
        }
    }

    /// Resets the decoder state, clearing any buffered partial character.
    #[inline]
    fn reset(&mut self) {
        self.len = 0;
        self.expected = 0;
    }
}

/// The ANSI lexer. Converts a byte stream into `AnsiToken`s.
///
/// Processes bytes individually, handles UTF-8 decoding, and manages state
/// across multiple calls to `process_byte`. Tokens can be retrieved using
/// `take_tokens`.
#[derive(Debug, Clone, Default)]
pub struct AnsiLexer {
    /// Buffer for completed tokens waiting to be consumed.
    tokens: Vec<AnsiToken>,
    /// Internal state machine for decoding UTF-8 characters.
    utf8_decoder: Utf8Decoder,
}

impl AnsiLexer {
    /// Creates a new `AnsiLexer` with default state.
    pub fn new() -> Self {
        AnsiLexer::default()
    }

    /// Processes a single byte, updating the internal token buffer and UTF-8 decoder state.
    ///
    /// - C0/C1 control codes interrupt any ongoing UTF-8 decoding and are emitted directly.
    /// - Other bytes are fed to the UTF-8 decoder. If a complete character (or replacement)
    ///   is decoded, a `Print` token is buffered.
    pub fn process_byte(&mut self, byte: u8) {
        match byte {
            // C0 Control Codes (excluding ESC) & DEL
            0x00..=0x1A | 0x1C..=0x1F | 0x7F => {
                // Interrupt and reset UTF-8 decoding if necessary
                if self.utf8_decoder.len > 0 {
                    self.tokens.push(AnsiToken::Print(REPLACEMENT_CHARACTER));
                    self.utf8_decoder.reset();
                }
                self.tokens.push(AnsiToken::C0Control(byte));
            }
            // Escape character
            0x1B => {
                // Interrupt and reset UTF-8 decoding if necessary
                if self.utf8_decoder.len > 0 {
                    self.tokens.push(AnsiToken::Print(REPLACEMENT_CHARACTER));
                    self.utf8_decoder.reset();
                }
                self.tokens.push(AnsiToken::C0Control(0x1B));
            }
            // C1 Control Codes
            0x80..=0x9F => {
                // Interrupt and reset UTF-8 decoding if necessary
                if self.utf8_decoder.len > 0 {
                    self.tokens.push(AnsiToken::Print(REPLACEMENT_CHARACTER));
                    self.utf8_decoder.reset();
                }
                self.tokens.push(AnsiToken::C1Control(byte));
            }
            // Potentially printable characters (ASCII or UTF-8)
            _ => {
                // Feed byte to the stateful decoder
                if let Some(c) = self.utf8_decoder.decode(byte) {
                    // A character (or replacement) was completed
                    self.tokens.push(AnsiToken::Print(c));
                }
                // If None, decoder is waiting for more bytes; state is preserved.
            }
        }
    }

    /// Takes the currently buffered tokens, leaving the internal buffer empty.
    ///
    /// Note: This does **not** reset the internal `Utf8Decoder` state if it's
    /// waiting for more bytes. This allows UTF-8 characters split across calls
    /// to `process_byte` to be decoded correctly.
    pub fn take_tokens(&mut self) -> Vec<AnsiToken> {
        mem::take(&mut self.tokens)
    }
}
