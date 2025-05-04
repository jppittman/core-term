// src/ansi/mod.rs

//! Central module for parsing ANSI escape sequences and representing them as commands.
//! This module contains the state machine for parsing and the definitions for the
//! different types of terminal commands that can be generated from the byte stream.

// Declare the sub-modules
mod parser;
mod commands;

// Re-export the main parser struct and the command enums for external use.
// This allows other parts of the crate (like the main Term struct) to
// use `use crate::ansi::{AnsiParser, AnsiCommand, ...};`
pub use parser::AnsiParser;
pub use commands::{AnsiCommand, C0Control, CsiCommand, OscCommand, SgrParameter, EraseMode, DecPrivateMode};


// --- Configuration Limits ---
// These define maximums for sequence components to prevent unbounded growth
// or denial-of-service attacks from malicious input.
// These are used by the parser and can be referenced here as configuration.
const MAX_CSI_PARAMS: usize = 16; // Maximum number of parameters in a CSI sequence
const MAX_CSI_INTERMEDIATES: usize = 2; // Maximum number of intermediate bytes in a CSI sequence
const MAX_OSC_STRING_LEN: usize = 256; // Maximum length of an OSC string in bytes

