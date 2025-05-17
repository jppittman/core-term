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

// Re-export items for easier use by other modules and within this module
pub use action::EmulatorAction;
pub use charset::{CharacterSet, map_to_dec_line_drawing};
pub use emulator::TerminalEmulator;
pub use modes::{DecModeConstant, DecPrivateModes, EraseMode, Mode};

// Crate-level imports (adjust paths based on where items are moved)
use crate::{
    ansi::commands::{
        AnsiCommand,
        Attribute,
        C0Control,
        Color as AnsiColor,
        CsiCommand,
        EscCommand, // Added EscCommand
    },
    backends::BackendEvent,
    color::{NamedColor, Color},
    glyph::{AttrFlags, Attributes, Glyph},
    term::cursor::{CursorController, ScreenContext},
    term::screen::Screen,
    term::unicode::get_char_display_width,
};

// Logging
use log::{debug, trace, warn};

/// Default tab interval.
pub const DEFAULT_TAB_INTERVAL: u8 = 8;
/// Default cursor shape (e.g., 2 for block).
const DEFAULT_CURSOR_SHAPE: u16 = 2;

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

/// Defines the interface the Renderer uses to interact with a terminal implementation.
pub trait TerminalInterface {
    fn dimensions(&self) -> (usize, usize);
    fn get_glyph(&self, x: usize, y: usize) -> Glyph;
    fn is_cursor_visible(&self) -> bool;
    fn get_screen_cursor_pos(&self) -> (usize, usize);
    fn take_dirty_lines(&mut self) -> Vec<usize>; // Needs &mut self
    fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction>;
}

// --- Implement TerminalInterface for TerminalEmulator ---
impl TerminalInterface for TerminalEmulator {
    fn dimensions(&self) -> (usize, usize) {
        (self.screen.width, self.screen.height)
    }

    /// Interprets an `EmulatorInput` and updates the terminal state.
    fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        let mut action = match input {
            EmulatorInput::Ansi(command) => self.handle_ansi_command(command),
            EmulatorInput::User(event) => self.handle_backend_event(event),
            EmulatorInput::RawChar(ch) => {
                self.print_char(ch);
                None
            }
        };

        if action.is_none() && !self.screen.dirty.iter().all(|&d| d == 0) {
            action = Some(EmulatorAction::RequestRedraw);
        }
        action
    }

    fn get_glyph(&self, x: usize, y: usize) -> Glyph {
        self.screen.get_glyph(x, y)
    }

    fn is_cursor_visible(&self) -> bool {
        // Visibility is determined by DECTCEM mode AND potentially blink state
        // For now, just the DECTCEM mode state.
        self.dec_modes.text_cursor_enable_mode
    }

    fn get_screen_cursor_pos(&self) -> (usize, usize) {
        self.cursor_controller
            .physical_screen_pos(&self.current_screen_context())
    }

    fn take_dirty_lines(&mut self) -> Vec<usize> {
        let mut all_dirty_indices: std::collections::HashSet<usize> =
            self.dirty_lines.drain(..).collect(); // Drains the legacy vec

        for (idx, &is_dirty_flag) in self.screen.dirty.iter().enumerate() {
            if is_dirty_flag != 0 {
                all_dirty_indices.insert(idx);
            }
        }
        self.screen.clear_dirty_flags();

        let mut sorted_dirty_lines: Vec<usize> = all_dirty_indices.into_iter().collect();
        sorted_dirty_lines.sort_unstable();
        sorted_dirty_lines
    }
}

#[cfg(test)]
mod tests;
