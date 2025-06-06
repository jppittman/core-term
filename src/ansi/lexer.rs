// src/ansi/lexer.rs

//! ANSI escape sequence lexer.
//! Converts a byte stream into `AnsiToken`s, processing byte by byte,
//! handling UTF-8 decoding and state across calls.

use log::{trace, warn};
use std::{mem, str};

/// Unicode replacement character (U+FFFD).
/// Used when encountering invalid UTF-8 sequences.
const REPLACEMENT_CHARACTER: char = '\u{FFFD}';

/// Represents the outcome of a single byte being processed by the Utf8Decoder.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Utf8DecodeResult {
    Decoded(char),   // Successfully decoded a valid Unicode character.
    InvalidSequence, // The byte sequence was invalid. Decoder is reset.
    NeedsMoreBytes,  // Current byte was validly consumed/buffered; more bytes needed.
}

// --- Constants for Control Code Ranges ---
// (These are from your provided code and look good for module-level constants)
const C0_CONTROL_PRINTABLE_PART1_RANGE: core::ops::RangeInclusive<u8> = 0x00..=0x1A; // NUL to SUB
const C0_CONTROL_PRINTABLE_PART2_RANGE: core::ops::RangeInclusive<u8> = 0x1C..=0x1F; // FS to US
const DEL_BYTE: u8 = 0x7F;
const ESC_BYTE: u8 = 0x1B;
const STRING_TERMINATOR: u8 = 0x9C;
const C1_CONTROL_RANGE: core::ops::RangeInclusive<u8> = 0x80..=0x9F;

// --- Constants for Unicode boundaries ---
const UNICODE_MAX_CODE_POINT: u32 = 0x10FFFF;
const UNICODE_SURROGATE_START: u32 = 0xD800;
const UNICODE_SURROGATE_END: u32 = 0xDFFF;

// --- Constants for UTF-8 byte classification (used by Utf8Decoder) ---
const UTF8_ASCII_MAX: u8 = 0x7F;
const UTF8_CONT_MIN: u8 = 0x80; // Start of continuation byte range
const UTF8_CONT_MAX: u8 = 0xBF; // End of continuation byte range
const UTF8_2_BYTE_MIN: u8 = 0xC2; // Excludes overlong 0xC0, 0xC1
                                  // const UTF8_2_BYTE_MAX: u8 = 0xDF; // Defined by next min - 1
const UTF8_3_BYTE_MIN: u8 = 0xE0;
// const UTF8_3_BYTE_MAX: u8 = 0xEF; // Defined by next min - 1
const UTF8_4_BYTE_MIN: u8 = 0xF0;
const UTF8_4_BYTE_MAX: u8 = 0xF4; // Max valid start for 4-byte sequence (RFC 3629)
const UTF8_INVALID_AS_START_MIN_RANGE1: u8 = UTF8_CONT_MIN; // 0x80 (can't start with continuation)
const UTF8_INVALID_AS_START_MAX_RANGE1: u8 = 0xC1; // Up to (and including) overlong 0xC1
const UTF8_INVALID_AS_START_MIN_RANGE2: u8 = 0xF5; // Invalid byte, per RFC 3629 (can't be > F4)
                                                   // const UTF8_INVALID_AS_START_MAX_RANGE2: u8 = 0xFF;       // Defined by u8 max

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
    buffer: [u8; 4],
    len: usize,
    expected: usize,
}

impl Utf8Decoder {
    #[inline]
    fn reset(&mut self) {
        self.len = 0;
        self.expected = 0;
    }

    fn decode(&mut self, byte: u8) -> Utf8DecodeResult {
        if self.len == 0 {
            return self.decode_first_byte(byte);
        }
        self.decode_continuation_byte(byte)
    }

    #[inline]
    fn decode_first_byte(&mut self, byte: u8) -> Utf8DecodeResult {
        let utf8_ascii_range: std::ops::RangeInclusive<u8> = 0x00..=UTF8_ASCII_MAX;
        let utf8_invalid_early_start_range: std::ops::RangeInclusive<u8> =
            UTF8_INVALID_AS_START_MIN_RANGE1..=UTF8_INVALID_AS_START_MAX_RANGE1; // 0x80..=0xC1
        let utf8_2_byte_start_range: std::ops::RangeInclusive<u8> = UTF8_2_BYTE_MIN..=0xDF;
        let utf8_3_byte_start_range: std::ops::RangeInclusive<u8> = UTF8_3_BYTE_MIN..=0xEF;
        let utf8_4_byte_start_range: std::ops::RangeInclusive<u8> =
            UTF8_4_BYTE_MIN..=UTF8_4_BYTE_MAX; // 0xF0..=0xF4
        let utf8_invalid_late_start_range: std::ops::RangeInclusive<u8> =
            UTF8_INVALID_AS_START_MIN_RANGE2..=0xFF; // 0xF5..=0xFF

        match byte {
            b if utf8_ascii_range.contains(&b) => Utf8DecodeResult::Decoded(b as char),
            b if utf8_2_byte_start_range.contains(&b) => {
                self.expected = 2;
                self.buffer[0] = b;
                self.len = 1;
                Utf8DecodeResult::NeedsMoreBytes
            }
            b if utf8_3_byte_start_range.contains(&b) => {
                self.expected = 3;
                self.buffer[0] = b;
                self.len = 1;
                Utf8DecodeResult::NeedsMoreBytes
            }
            b if utf8_4_byte_start_range.contains(&b) => {
                self.expected = 4;
                self.buffer[0] = b;
                self.len = 1;
                Utf8DecodeResult::NeedsMoreBytes
            }
            // Catches invalid start bytes: 0x80-0xC1 (continuation / overlong C0/C1) and 0xF5-0xFF
            b if utf8_invalid_early_start_range.contains(&b)
                || utf8_invalid_late_start_range.contains(&b) =>
            {
                warn!("invalid utf8 sequence byte: {:X?}", b);
                self.reset();
                Utf8DecodeResult::InvalidSequence
            }
            _ => {
                unreachable!(
                    "This default branch should ideally not be hit if ranges cover all u8 values."
                );
            }
        }
    }

    #[inline]
    fn decode_continuation_byte(&mut self, byte: u8) -> Utf8DecodeResult {
        let utf8_continuation_range: std::ops::RangeInclusive<u8> = UTF8_CONT_MIN..=UTF8_CONT_MAX; // 0x80..=0xBF

        if !utf8_continuation_range.contains(&byte) {
            // Current `byte` is not a valid UTF-8 continuation.
            // The previously buffered sequence is now considered invalid.
            self.reset();
            return Utf8DecodeResult::InvalidSequence;
        }

        self.buffer[self.len] = byte;
        self.len += 1;

        if self.len != self.expected {
            return Utf8DecodeResult::NeedsMoreBytes;
        }

        let b = self.buffer;
        // Sequence is now notionally complete, try to make a char.
        // This also implicitly handles overlong sequences if `str::from_utf8` is strict.
        let char_str_result = str::from_utf8(&b[0..self.len]);
        self.reset();

        match char_str_result {
            Ok(s) => {
                // `from_utf8` guarantees `s` is valid UTF-8.
                // A multi-byte UTF-8 sequence will produce exactly one char.
                if let Some(c) = s.chars().next() {
                    let cp = c as u32;
                    // Final check for Unicode constraints (surrogates, max codepoint).
                    if cp <= UNICODE_MAX_CODE_POINT
                        && !(UNICODE_SURROGATE_START..=UNICODE_SURROGATE_END).contains(&cp)
                    {
                        Utf8DecodeResult::Decoded(c)
                    } else {
                        Utf8DecodeResult::InvalidSequence // Valid UTF-8 but invalid Unicode scalar value
                    }
                } else {
                    // Should be impossible for non-empty slice that `from_utf8` said was Ok.
                    Utf8DecodeResult::InvalidSequence
                }
            }
            Err(_) => Utf8DecodeResult::InvalidSequence, // Malformed UTF-8 byte sequence.
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct AnsiLexer {
    tokens: Vec<AnsiToken>,
    utf8_decoder: Utf8Decoder,
}

impl AnsiLexer {
    pub fn new() -> Self {
        AnsiLexer::default()
    }

    /// Determines if a byte is a C0 (excluding NUL, HT, LF, CR, etc. if meant as data),
    /// ESC, or DEL that should unambiguously interrupt any ongoing UTF-8 sequence.
    /// C1 codes are *not* checked here because their byte values can be valid UTF-8 continuations;
    /// they are handled by the Utf8Decoder returning InvalidSequence if they break a sequence.
    #[inline]
    fn is_unambiguous_interrupting_control(byte: u8) -> bool {
        byte == ESC_BYTE || byte == STRING_TERMINATOR ||
        // Consider which C0s truly interrupt. For now, all except common formatting.
        // Some tests might expect NUL or other C0s to also interrupt.
        // This list should match C0s that are *never* valid data mid-UTF-8.
        match byte {
            // Explicitly allow common formatting C0s to pass through to the decoder
            // if they are not part of a multi-byte sequence (handled by decoder).
            // However, if UTF-8 is in progress, ANY C0 is an interruption.
            0x00..=0x08 | 0x0B..=0x0C | 0x0E..=0x1A | 0x1C..=0x1F | DEL_BYTE => true,
            _ => false,
        }
    }

    /// Determines if a byte is any C0, ESC, C1, or DEL control code.
    /// Used when processing a byte from a ground state (no active UTF-8 sequence).
    #[inline]
    fn is_any_control_code(byte: u8) -> bool {
        (C0_CONTROL_PRINTABLE_PART1_RANGE.contains(&byte))
            || (C0_CONTROL_PRINTABLE_PART2_RANGE.contains(&byte))
            || byte == DEL_BYTE
            || byte == ESC_BYTE
            || C1_CONTROL_RANGE.contains(&byte)
    }

    fn process_byte_as_new_token(&mut self, byte: u8) {
        // This function is called when utf8_decoder.len == 0.
        // It decides if 'byte' is a control code or starts a new UTF-8 sequence.
        if Self::is_any_control_code(byte) {
            match byte {
                ESC_BYTE => self.tokens.push(AnsiToken::C0Control(ESC_BYTE)),
                b if C1_CONTROL_RANGE.contains(&b) => { /* Do nothing, ignore C1 control per plan */ }
                // All other C0s (including those in C0_CONTROL_PRINTABLE_PART1/2 and DEL_BYTE)
                b => self.tokens.push(AnsiToken::C0Control(b)), // Catches all remaining C0s and DEL
            }
        } else {
            // Not a control code, so attempt to process as UTF-8 start.
            // Utf8Decoder is fresh (len == 0).
            match self.utf8_decoder.decode(byte) {
                Utf8DecodeResult::Decoded(c) => self.tokens.push(AnsiToken::Print(c)),
                Utf8DecodeResult::NeedsMoreBytes => { /* Byte buffered, wait for more */ }
                Utf8DecodeResult::InvalidSequence => {
                    // This means 'byte' itself was an invalid UTF-8 start (e.g., 0xC0, 0xF5).
                    warn!(
                        "invalid utf8 byte: {:X?} printing replacment character",
                        byte
                    );
                    self.tokens.push(AnsiToken::Print(REPLACEMENT_CHARACTER));
                }
            }
        }
    }

    pub fn process_byte(&mut self, byte: u8) {
        if self.utf8_decoder.len > 0 {
            // --- Currently building a multi-byte UTF-8 char ---
            // Check for unambiguous interruptions (ESC, most C0s)
            if Self::is_unambiguous_interrupting_control(byte) {
                warn!("encountered control byte: {:X?} mid utf8 stream", byte);
                self.tokens.push(AnsiToken::Print(REPLACEMENT_CHARACTER)); // For the aborted UTF-8
                self.utf8_decoder.reset();
                self.process_byte_as_new_token(byte); // Process the interrupting C0/ESC
                return;
            }

            // If `byte` is a C1 code (0x80-0x9F), it was NOT caught by unambiguous_interrupting_control.
            // Let the Utf8Decoder try to process it. If it's not a valid continuation for
            // the current sequence, Utf8Decoder will return InvalidSequence.
            match self.utf8_decoder.decode(byte) {
                Utf8DecodeResult::Decoded(c) => {
                    self.tokens.push(AnsiToken::Print(c));
                    // Decoder has reset.
                }
                Utf8DecodeResult::InvalidSequence => {
                    // `byte` (which could be a C1, or non-control like 'A')
                    // was not a valid continuation for what was in the buffer.
                    // Utf8Decoder has reset.
                    self.tokens.push(AnsiToken::Print(REPLACEMENT_CHARACTER)); // For the broken sequence
                                                                               // Now, reprocess `byte` from a ground state.
                                                                               // process_byte_as_new_token will correctly identify it if it's C1, C0, ESC, or data.
                    self.process_byte_as_new_token(byte);
                }
                Utf8DecodeResult::NeedsMoreBytes => {
                    // Valid continuation, byte buffered. Wait for more.
                }
            }
        } else {
            // --- Not currently building a multi-byte char (utf8_decoder.len == 0) ---
            self.process_byte_as_new_token(byte);
        }
    }

    pub fn take_tokens(&mut self) -> Vec<AnsiToken> {
        trace!("taking {:?} tokens from lexer", &self.tokens);
        mem::take(&mut self.tokens)
    }

    /// Finalizes any incomplete UTF-8 sequence, e.g., at end of stream.
    /// This is called by the AnsiProcessor after processing a chunk of bytes.
    pub fn finalize(&mut self) {
        if self.utf8_decoder.len > 0 {
            self.tokens.push(AnsiToken::Print(REPLACEMENT_CHARACTER));
            self.utf8_decoder.reset();
        }
    }
}
