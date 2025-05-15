// src/term/input.rs

//! Defines the types of input that the terminal emulator can process.
//! These inputs drive the terminal's state machine, encompassing both
//! data from the underlying program (like ANSI commands) and interactions
//! from the user via the display backend.

// Necessary imports for the types used within EmulatorInput.
// These paths assume that when input.rs is part of the term module,
// these crate-level paths will be valid.
use crate::ansi::commands::AnsiCommand;
use crate::backends::BackendEvent;

/// Inputs that the terminal emulator processes.
///
/// This enum encapsulates the different kinds of data or events
/// that the `TerminalEmulator` can receive and act upon. It serves as the
/// primary "instruction set" for the terminal's internal state machine.
#[derive(Debug, Clone)]
pub enum EmulatorInput {
    /// An ANSI command or sequence parsed from the output of the
    /// program running in the PTY (Pseudo-Terminal).
    Ansi(AnsiCommand),

    /// An event originating from the user (e.g., keyboard input) or the
    /// backend system (e.g., window resize, focus change), as reported
    /// by the `Driver`.
    User(BackendEvent),

    /// A single raw character. This variant might be used for scenarios
    /// where direct character printing is intended without full ANSI
    /// processing, or for specific unhandled cases. (Consider if this
    /// should always be wrapped in an AnsiCommand::Print for consistency).
    RawChar(char),
}
