// src/ansi/mod.rs

//! Handles ANSI escape sequence parsing.

mod commands;
mod lexer;
mod parser;
// mod tests; // Keep tests separate if using #[cfg(test)]

// Re-export necessary items for public API
pub use commands::AnsiCommand; // Keep AnsiCommand public
// Removed unused re-exports: CsiCommand, C0Control
use lexer::{AnsiLexer, AnsiToken};
use parser::AnsiParser;

/// The main processor that combines the lexer and parser.
/// It takes byte slices as input and provides parsed commands.
#[derive(Debug, Default)]
pub struct AnsiProcessor {
    lexer: AnsiLexer,
    parser: AnsiParser,
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

    // Note: `take_commands` is now called directly on `processor.parser` in tests.
    // If needed publicly, add a wrapper method here:
    // pub fn take_commands(&mut self) -> Vec<AnsiCommand> {
    //     self.parser.take_commands()
    // }
}

// Include tests module if defined in this file
#[cfg(test)]
mod tests; // Assuming tests are in ansi/tests.rs


