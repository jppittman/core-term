// src/ansi/mod.rs

//! Handles ANSI escape sequence parsing.

// Make submodules public so their contents can be used elsewhere
pub mod commands;
mod lexer;
mod parser;
// Re-export necessary items for public API
pub use commands::AnsiCommand;

// Keep internal components private to this module unless needed outside
use lexer::AnsiLexer;
use parser::AnsiParser as ParserImpl;

pub trait AnsiParser {
    fn process_bytes(&mut self, bytes: &[u8]) -> Vec<AnsiCommand>;
}

/// The main processor that combines the lexer and parser.
/// It takes byte slices as input and provides parsed commands.
#[derive(Debug, Default)]
pub struct AnsiProcessor {
    pub(super) lexer: AnsiLexer,
    pub(super) parser: ParserImpl,
}

impl AnsiProcessor {
    /// Creates a new `AnsiProcessor`.
    pub fn new() -> Self {
        AnsiProcessor {
            lexer: AnsiLexer::new(),
            parser: ParserImpl::new(),
        }
    }
}

impl AnsiParser for AnsiProcessor {
    /// Processes a slice of bytes.
    ///
    /// Bytes are lexed into tokens, and tokens are processed by the parser.
    /// Call `take_commands` on the `parser` field to retrieve results.
    fn process_bytes(&mut self, bytes: &[u8]) -> Vec<AnsiCommand> {
        for byte in bytes {
            self.lexer.process_byte(*byte);
        }
        // Finalize any pending UTF-8 sequence in the lexer.
        self.lexer.finalize();

        // Now take all tokens, including any finalization token.
        let tokens = self.lexer.take_tokens();
        for token in tokens {
            self.parser.process_token(token);
        }
        self.parser.take_commands()
    }
}

// Include tests module if defined in this file
#[cfg(test)]
mod tests; // Assuming tests are in ansi/tests.rs
