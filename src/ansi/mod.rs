// src/ansi/mod.rs

// Declare the internal modules within the ansi package.
mod lexer;
mod parser;
mod commands;

// Re-export key types that are part of the public API.
// Code using the ansi module will primarily interact with these commands.
pub use commands::{AnsiCommand, CsiCommand, C0Control};

// Import the internal lexer and parser structs.
use lexer::AnsiLexer;
use parser::AnsiParser;

/// A public struct that combines the AnsiLexer and AnsiParser
/// to process a stream of bytes and produce a list of AnsiCommands.
/// This is the main entry point for using the ANSI parsing functionality.
pub struct AnsiProcessor {
    lexer: AnsiLexer,
    parser: AnsiParser,
}

impl AnsiProcessor {
    /// Creates a new `AnsiProcessor`.
    /// Initializes the internal lexer and parser.
    pub fn new() -> Self {
        AnsiProcessor {
            lexer: AnsiLexer::new(),
            parser: AnsiParser::new(),
        }
    }

    /// Processes a slice of bytes containing ANSI escape sequences and characters.
    ///
    /// This method feeds the bytes through the internal lexer to produce tokens,
    /// and then feeds the tokens through the internal parser to produce a list
    /// of structured `AnsiCommand`s.
    ///
    /// The internal state of the lexer and parser is maintained between calls,
    /// allowing for processing of fragmented ANSI sequences across multiple `process_bytes` calls.
    ///
    /// # Arguments
    ///
    /// * `bytes`: A slice of bytes to process.
    ///
    /// # Returns
    ///
    /// A `Vec<AnsiCommand>` containing the parsed commands from the input bytes.
    pub fn process_bytes(&mut self, bytes: &[u8]) -> Vec<AnsiCommand> {
        // Process each byte with the lexer.
        // The lexer updates its internal state and token buffer.
        for &byte in bytes {
            self.lexer.process_byte(byte);
        }

        // Take all tokens that the lexer has produced so far.
        let tokens = self.lexer.take_tokens();

        // Process each token with the parser.
        // The parser updates its internal state and command buffer.
        for token in tokens {
            self.parser.process_token(token);
        }

        // Take all commands that the parser has produced so far.
        // These are the fully formed ANSI commands ready to be interpreted
        // by a terminal emulator or application logic.
        self.parser.take_commands()
    }

    // Add other public methods here if needed, e.g., to process a single byte,
    // reset the processor state, or get the current state for debugging/inspection.
}

#[cfg(test)]
mod tests;

