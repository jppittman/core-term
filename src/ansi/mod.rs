// src/ansi/mod.rs

//! Handles ANSI escape sequence parsing.

mod commands;
mod lexer;
mod parser;

// Re-export necessary items for public API
 // Keep AnsiCommand public
// Removed unused re-exports: CsiCommand, C0Control
// Also remove AnsiToken if not used publicly
use lexer::AnsiLexer; // Keep AnsiLexer private if only used here
use parser::AnsiParser; // Keep AnsiParser private

/// The main processor that combines the lexer and parser.
/// It takes byte slices as input and provides parsed commands.
#[derive(Debug, Default)]
pub struct AnsiProcessor {
    lexer: AnsiLexer,
    // Make parser public if tests need direct access
    pub parser: AnsiParser,
}

impl AnsiProcessor {
    /// Creates a new `AnsiProcessor`.
    pub fn new() -> Self {
        AnsiProcessor {
            lexer: AnsiLexer::new(),
            parser: AnsiParser::new(),
        }
    }

    /// Processes a slice of bytes.
    ///
    /// Bytes are lexed into tokens, and tokens are processed by the parser.
    /// Call `take_commands` on the `parser` field to retrieve results.
    pub fn process_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.lexer.process_byte(*byte);
        }
        let tokens = self.lexer.take_tokens();
        for token in tokens {
            self.parser.process_token(token);
        }
    }
}

// Include tests module if defined in this file
#[cfg(test)]
mod tests; // Assuming tests are in ansi/tests.rs


