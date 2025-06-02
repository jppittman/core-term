// myterm/src/term/mod.rs

//! This module defines the core terminal emulator logic.
//! It acts as a state machine processing inputs and producing actions.

// Sub-modules - existing and new
pub mod cursor;
pub mod screen;
pub mod unicode;

pub mod action;
pub mod charset;
mod emulator;
pub mod modes;
pub mod snapshot; // Add this line to declare the module

// Re-export items for easier use by other modules and within this module
pub use action::{ControlEvent, EmulatorAction, UserInputAction}; // Added UserInputAction, ControlEvent
pub use charset::{map_to_dec_line_drawing, CharacterSet};
pub use emulator::TerminalEmulator;
pub use modes::{DecModeConstant, DecPrivateModes, EraseMode, Mode};
pub use snapshot::{
    CursorRenderState,
    CursorShape,
    Point,
    RenderSnapshot,
    Selection, // Changed SelectionRenderState to Selection
    SelectionMode,
    SnapshotLine,
};

// Crate-level imports (adjust paths based on where items are moved)
use crate::ansi::commands::AnsiCommand;
// Explicitly import Color and NamedColor if they are used directly in this module's functions,
// though they are mostly encapsulated within other types like Attributes.

// Logging

/// Default tab interval.
pub const DEFAULT_TAB_INTERVAL: u8 = 8;

// ControlEvent is now defined in and re-exported from action.rs

/// Inputs that the terminal emulator processes.
///
/// This enum encapsulates the different kinds of data or events
/// that the `TerminalEmulator` can receive and act upon. It serves as the
/// primary "instruction set" for the terminal's internal state machine.
#[derive(Debug, Clone, PartialEq)]
pub enum EmulatorInput {
    /// An ANSI command or sequence parsed from the output of the
    /// program running in the PTY (Pseudo-Terminal).
    Ansi(AnsiCommand),

    /// An event originating from the user (e.g., keyboard input) or the
    /// backend system (e.g., window resize, focus change), as reported
    /// by the `Driver` and translated by the `AppOrchestrator`.
    User(UserInputAction),
    /// An internal control event, such as a resize notification from the orchestrator.
    Control(ControlEvent), // Uses ControlEvent from action.rs

    /// A single raw character. This variant might be used for scenarios
    /// where direct character printing is intended without full ANSI
    /// processing, or for specific unhandled cases. (Consider if this
    /// should always be wrapped in an AnsiCommand::Print for consistency).
    RawChar(char),
}

/// Defines the essential public interface for a terminal emulator.
/// This interface is used by components like the `AppOrchestrator` and `Renderer`.
pub trait TerminalInterface {
    /// Interprets an `EmulatorInput` and updates the terminal state.
    /// Returns an `Option<EmulatorAction>` if the input results in an action
    /// that needs to be handled externally.
    fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction>;

    /// Creates a snapshot of the terminal's current visible state for rendering.
    /// This method provides all necessary information for the renderer to draw
    /// the terminal screen, including dirty flags, cell data, cursor, and selection.
    fn get_render_snapshot(&self) -> RenderSnapshot;
}

impl TerminalInterface for TerminalEmulator {
    /// Interprets an `EmulatorInput` and updates the terminal state.
    fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        // This just calls the inherent method on TerminalEmulator (defined in src/term/emulator/mod.rs)
        self.interpret_input(input)
    }

    /// Creates a snapshot of the terminal's current visible state for rendering.
    fn get_render_snapshot(&self) -> RenderSnapshot {
        // This just calls the inherent method on TerminalEmulator (defined in src/term/emulator/mod.rs)
        self.get_render_snapshot()
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod core_tests;
