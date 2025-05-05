// src/ansi/mod.rs

//! Handles ANSI escape sequence parsing.

// Make submodules public so their contents can be used elsewhere
pub mod commands;
pub mod lexer;
pub mod parser;

// Re-export necessary items for public API
pub use commands::AnsiCommand; // Keep AnsiCommand public

// Keep internal components private to this module unless needed outside
use lexer::AnsiLexer;
use parser::AnsiParser;

/// The main processor that combines the lexer and parser.
/// It takes byte slices as input and provides parsed commands.
#[derive(Debug, Default)]
pub struct AnsiProcessor {
    lexer: AnsiLexer,
    // Make parser public if tests need direct access to take_commands
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

