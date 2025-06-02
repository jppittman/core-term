// src/term/emulator/mod.rs

//! Core terminal emulation logic and state management.
//!
//! This module defines the `TerminalEmulator` struct, which is the heart of the
//! terminal processing. It manages screen state, cursor, character sets, modes,
//! and handles input (both from PTY and user) to update the terminal state
//! and generate actions for the orchestrator.

// Crate-level imports
use crate::{
    glyph::Attributes, // Ensure Glyph and Attributes are imported
    term::{
        action::{
            EmulatorAction,
            // MouseButton, // Unused
            // MouseEventType, // Unused
            // UserInputAction, // Unused
        },
        charset::CharacterSet,
        cursor::{self, CursorController, ScreenContext}, // Import cursor module for its CursorShape
        modes::DecPrivateModes,
        screen::Screen,
        snapshot::{
            CursorRenderState,
            CursorShape,
            Point, // Unused
            RenderSnapshot,
            SelectionMode, // Unused
            SnapshotLine,
        },
        EmulatorInput, // Added EmulatorInput
    },
};

// Logging (optional, but good practice if used)
use log::debug;

mod ansi_handler;
mod char_processor;
mod cursor_handler;
mod input_handler;
mod methods;
mod mode_handler;
mod osc_handler;
mod screen_ops;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum FocusState {
    Focused,
    Unfocused,
}

/// The core terminal emulator.
#[derive(Clone)]
pub struct TerminalEmulator {
    pub(super) screen: Screen,
    pub(super) focus_state: FocusState,
    pub(super) cursor_controller: CursorController,
    pub(super) dec_modes: DecPrivateModes,
    pub(super) active_charsets: [CharacterSet; 4],
    pub(super) active_charset_g_level: usize,
    pub(super) cursor_wrap_next: bool,
}

impl TerminalEmulator {
    /// Creates a new `TerminalEmulator`.
    pub fn new(width: usize, height: usize) -> Self {
        // Removed scrollback_limit parameter
        let initial_attributes = Attributes::default(); // SGR Reset attributes
                                                        // Screen::new now gets scrollback_limit from CONFIG
        let mut screen = Screen::new(width, height);
        // Ensure the screen's default_attributes are initialized correctly.
        // This is crucial for clearing operations.
        // Note: Screen::new now also initializes default_attributes from CONFIG.
        // This line might be redundant or ensure a specific SGR reset state if different.
        // For now, keeping it to ensure SGR reset state is applied if it differs from CONFIG's default.
        screen.default_attributes = initial_attributes;

        TerminalEmulator {
            screen,
            cursor_controller: CursorController::new(initial_attributes),
            dec_modes: DecPrivateModes::default(),
            active_charsets: [
                CharacterSet::Ascii, // G0
                CharacterSet::Ascii, // G1
                CharacterSet::Ascii, // G2
                CharacterSet::Ascii, // G3
            ],
            focus_state: FocusState::Focused,
            active_charset_g_level: 0, // Default to G0
            cursor_wrap_next: false,
        }
    }

    /// Helper to create the current `ScreenContext` for `CursorController`.
    pub(super) fn current_screen_context(&self) -> ScreenContext {
        ScreenContext {
            width: self.screen.width,
            height: self.screen.height,
            scroll_top: self.screen.scroll_top(),
            scroll_bot: self.screen.scroll_bot(),
            origin_mode_active: self.dec_modes.origin_mode,
        }
    }

    pub(super) fn dimensions(&self) -> (usize, usize) {
        (self.screen.width, self.screen.height)
    }

    /// Resizes the terminal display grid.
    pub(super) fn resize(&mut self, cols: usize, rows: usize) {
        self.cursor_wrap_next = false;
        // Screen::resize now gets scrollback_limit from CONFIG
        self.screen.resize(cols, rows);
        let (log_x, log_y) = self.cursor_controller.logical_pos();
        self.cursor_controller
            .move_to_logical(log_x, log_y, &self.current_screen_context());
        debug!(
            "Terminal resized to {}x{}. Cursor re-clamped. All lines marked dirty by screen.resize().",
            cols, rows
        );
    }

    /// Interprets a given `EmulatorInput` and updates terminal state.
    /// Returns an `Option<EmulatorAction>` if the input results in an action
    /// that needs to be handled externally (e.g., writing to PTY).
    pub fn interpret_input(&mut self, input: EmulatorInput) -> Option<EmulatorAction> {
        match input {
            EmulatorInput::Ansi(command) => {
                // Delegate to ANSI command handler
                ansi_handler::process_ansi_command(self, command)
            }
            EmulatorInput::User(action) => {
                // Delegate to user input action handler
                input_handler::process_user_input_action(self, action)
            }
            EmulatorInput::Control(event) => {
                // Delegate to control event handler
                input_handler::process_control_event(self, event)
            }
            EmulatorInput::RawChar(ch) => {
                // Delegate to raw character processor
                self.print_char(ch);
                None
            }
        }
    }

    /// Creates a snapshot of the terminal's current visible state for rendering.
    pub fn get_render_snapshot(&self) -> RenderSnapshot {
        let (width, height) = (self.screen.width, self.screen.height);
        let mut lines = Vec::with_capacity(height);
        let active_grid = self.screen.active_grid();

        for y_idx in 0..height {
            // Screen lines are directly 0..height for the visible part in active_grid
            let line_glyphs = active_grid[y_idx].clone();
            let is_dirty = self.screen.dirty.get(y_idx).map_or(true, |&d| d != 0); // Mark dirty if out of bounds or non-zero
            lines.push(SnapshotLine {
                is_dirty,
                cells: line_glyphs,
            });
        }

        let mut cursor_state = None;
        // Check DECTCEM mode for cursor visibility
        if self.dec_modes.text_cursor_enable_mode {
            let (cursor_x, cursor_y) = self
                .cursor_controller
                .physical_screen_pos(&self.current_screen_context());

            let (cell_char_underneath, cell_attributes_underneath) =
                if cursor_y < height && cursor_x < width {
                    // Use active_grid directly as get_glyph might have side effects or different logic
                    let glyph_wrapper = &active_grid[cursor_y][cursor_x];
                    match glyph_wrapper {
                        crate::glyph::Glyph::Single(cell)
                        | crate::glyph::Glyph::WidePrimary(cell) => (cell.c, cell.attr),
                        crate::glyph::Glyph::WideSpacer { .. } => {
                            (crate::glyph::WIDE_CHAR_PLACEHOLDER, Attributes::default())
                        }
                    }
                } else {
                    (' ', Attributes::default()) // Default if cursor is out of bounds
                };

            // Map internal cursor_controller.cursor.shape to term::snapshot::CursorShape
            let internal_shape = self.cursor_controller.cursor.shape;
            let mapped_shape = match internal_shape {
                cursor::CursorShape::BlinkingBlock | cursor::CursorShape::SteadyBlock => {
                    CursorShape::Block
                }
                cursor::CursorShape::BlinkingUnderline | cursor::CursorShape::SteadyUnderline => {
                    CursorShape::Underline
                }
                cursor::CursorShape::BlinkingBar | cursor::CursorShape::SteadyBar => {
                    CursorShape::Bar
                }
            };

            cursor_state = Some(CursorRenderState {
                x: cursor_x,
                y: cursor_y,
                shape: mapped_shape,
                cell_char_underneath,
                cell_attributes_underneath,
            });
        }

        RenderSnapshot {
            dimensions: (width, height),
            lines,
            cursor_state,
            selection: self.screen.selection.clone(),
        }
    }

    // --- Selection Handling Methods ---

    pub fn start_selection(&mut self, point: Point, mode: SelectionMode) {
        self.screen.start_selection(point, mode);
        debug!(
            "Selection started at ({}, {}) with mode {:?}",
            point.x, point.y, mode
        );
    }

    pub fn extend_selection(&mut self, point: Point) {
        self.screen.update_selection(point);
        debug!("Selection extended to ({}, {})", point.x, point.y);
    }

    pub fn apply_selection_clear(&mut self) {
        if self.screen.selection.is_active {
            if let Some(range) = self.screen.selection.range {
                if range.start == range.end {
                    self.screen.clear_selection();
                    debug!("Selection applied (was a click): cleared.");
                } else {
                    self.screen.selection.is_active = false;
                    debug!(
                        "Selection applied (was a drag): finalized at range {:?}.",
                        range
                    );
                }
            } else {
                self.screen.selection.is_active = false;
                debug!("Selection applied (no range): cleared active state.");
            }
        }
    }

    pub fn clear_selection(&mut self) {
        self.screen.clear_selection();
        debug!("Selection cleared.");
    }

    pub fn get_selected_text(&self) -> Option<String> {
        self.screen.get_selected_text()
    }

    pub fn paste_text(&mut self, text: String) {
        if self.dec_modes.bracketed_paste_mode {
            log::warn!("TerminalEmulator::paste_text called with bracketed paste mode ON. This mode should be handled by the caller (input_handler) by wrapping the text and sending it as WritePty. Processing char by char as fallback.");
            for ch in text.chars() {
                self.print_char(ch);
            }
        } else {
            log::debug!(
                "TerminalEmulator::paste_text - Bracketed Paste Mode OFF. Processing {} chars.",
                text.len()
            );
            for ch in text.chars() {
                self.print_char(ch);
            }
        }
    }
}
