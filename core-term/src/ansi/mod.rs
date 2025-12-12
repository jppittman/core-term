// src/ansi/mod.rs

//! Provides a parser for ANSI escape sequences.
//!
//! This module contains the `AnsiParser` trait, which defines the core interface
//! for parsing byte streams into `AnsiCommand`s, and its primary implementation,
//! `AnsiProcessor`. The parser is designed to be stateful and can process
//! input incrementally.

pub mod commands;
mod lexer;
pub mod parser;

pub use commands::AnsiCommand;
use lexer::AnsiLexer;
use parser::AnsiParser as ParserImpl;

/// A trait for stateful ANSI escape sequence parsers.
///
/// This trait defines the essential functionality for a parser that consumes
/// bytes and produces a sequence of `AnsiCommand`s.
pub trait AnsiParser {
    /// Processes a byte slice and returns any newly parsed commands.
    ///
    /// This method feeds the given bytes into the parser's state machine.
    /// The parser will consume the bytes, update its internal state, and return
    /// a vector of any `AnsiCommand`s that were completed during this chunk
    /// of processing.
    ///
    /// # Arguments
    ///
    /// * `bytes` - A slice of bytes to be processed.
    fn process_bytes(&mut self, bytes: &[u8]) -> Vec<AnsiCommand>;
}

/// A stateful processor that parses a stream of bytes into `AnsiCommand`s.
///
/// This struct implements the `AnsiParser` trait and serves as the main
/// entry point for ANSI parsing. It internally uses a lexer and a parser
/// to transform the raw byte stream into structured commands.
#[derive(Debug, Default)]
pub struct AnsiProcessor {
    pub(super) lexer: AnsiLexer,
    pub(super) parser: ParserImpl,
}

impl AnsiProcessor {
    /// Creates a new, default `AnsiProcessor`.
    pub fn new() -> Self {
        AnsiProcessor {
            lexer: AnsiLexer::new(),
            parser: ParserImpl::new(),
        }
    }

    /// Advances the parser with a single byte and returns a command if one is complete.
    /// Used for byte-by-byte processing in streaming contexts.
    pub fn advance(&mut self, byte: u8) -> Option<AnsiCommand> {
        self.lexer.process_byte(byte);
        // Take tokens immediately
        let tokens = self.lexer.take_tokens();
        for token in tokens {
            self.parser.process_token(token);
        }
        // Return first command if any (simplified)
        self.parser.take_commands().pop()
    }
}

impl AnsiParser for AnsiProcessor {
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
