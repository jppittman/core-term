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
            // Point, // Unused
            RenderSnapshot,
            // SelectionMode, // Unused
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
    pub fn new(width: usize, height: usize, scrollback_limit: usize) -> Self {
        let initial_attributes = Attributes::default(); // SGR Reset attributes
        let mut screen = Screen::new(width, height, scrollback_limit);
        // Ensure the screen's default_attributes are initialized correctly.
        // This is crucial for clearing operations.
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
        let current_scrollback_limit = self.screen.scrollback_limit();
        self.screen.resize(cols, rows, current_scrollback_limit);
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
                    let glyph = &active_grid[cursor_y][cursor_x];
                    (glyph.c, glyph.attr)
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

        // Populate selection state
        let selection_state = if self.screen.selection.start.is_some() {
            Some(self.screen.selection)
        } else {
            None
        };

        RenderSnapshot {
            dimensions: (width, height),
            lines,
            cursor_state,
            selection: self.screen.selection, // Updated to use the non-optional field
        }
    }

    // --- Selection Handling Methods ---

    /// Starts a new selection at the given screen cell coordinates.
    ///
    /// Args:
    /// * `point`: The `Point` (column, row) where the selection starts.
    pub fn start_selection(&mut self, point: crate::term::snapshot::Point) {
        self.screen.selection.range = Some(crate::term::snapshot::SelectionRange {
            start: point,
            end: point,
        });
        self.screen.selection.is_active = true;
        self.screen.selection.mode = crate::term::snapshot::SelectionMode::Cell; // Default to cell selection
        // Mark relevant lines as dirty if needed for rendering updates
        // This might be handled by a more general redraw request.
        debug!("Selection started at ({}, {})", point.x, point.y);
    }

    /// Extends the currently active selection to the given screen cell coordinates.
    ///
    /// If no selection is active, this function does nothing.
    ///
    /// Args:
    /// * `point`: The `Point` (column, row) to extend the selection to.
    pub fn extend_selection(&mut self, point: crate::term::snapshot::Point) {
        if self.screen.selection.is_active {
            if let Some(ref mut range) = self.screen.selection.range {
                range.end = point;
                // Mark relevant lines as dirty
                debug!("Selection extended to ({}, {})", point.x, point.y);
            }
        }
    }

    /// Finalizes the current selection (e.g., on mouse button release).
    ///
    /// If the selection start and end points are the same (indicating a click
    /// without dragging), the selection is cleared. Otherwise, the selection
    /// is marked as inactive but preserved.
    pub fn apply_selection_clear(&mut self) {
        if self.screen.selection.is_active {
            if let Some(range) = self.screen.selection.range {
                if range.start == range.end {
                    // This was a click, not a drag. Clear the selection.
                    self.clear_selection();
                    debug!("Selection applied (was a click): cleared.");
                } else {
                    // This was a drag. Finalize selection.
                    self.screen.selection.is_active = false;
                    debug!("Selection applied (was a drag): finalized at range {:?}.", range);
                }
            } else {
                // No range, but was active? Clear active state.
                self.screen.selection.is_active = false;
                debug!("Selection applied (no range): cleared active state.");
            }
        }
    }

    /// Clears any existing selection and marks it as inactive.
    pub fn clear_selection(&mut self) {
        self.screen.selection.range = None;
        self.screen.selection.is_active = false;
        // Mark previously selected lines as dirty if needed for rendering updates
        debug!("Selection cleared.");
    }

    /// Retrieves the text currently selected in the terminal.
    ///
    /// The selection is determined by `self.screen.selection.range`.
    /// Text is collected from the active screen buffer.
    /// Lines are separated by `\n`. Trailing whitespace on lines within the
    /// selection is generally preserved, especially if the selection spans
    /// multiple lines or extends to the end of a line.
    pub fn get_selected_text(&self) -> Option<String> {
        let selection_range = self.screen.selection.range?; // Returns None if no range

        // Determine normalized start and end points (top-left to bottom-right)
        let (start_point, end_point) = if selection_range.start.y < selection_range.end.y ||
                                          (selection_range.start.y == selection_range.end.y && selection_range.start.x <= selection_range.end.x) {
            (selection_range.start, selection_range.end)
        } else {
            (selection_range.end, selection_range.start)
        };

        let mut selected_text = String::new();
        let active_grid = self.screen.active_grid();
        let (cols, _rows) = self.dimensions();

        for y_idx in start_point.y..=end_point.y {
            if y_idx >= active_grid.len() { // Should not happen if selection is valid
                continue;
            }
            let line = &active_grid[y_idx];

            let start_x = if y_idx == start_point.y { start_point.x } else { 0 };
            let end_x = if y_idx == end_point.y { end_point.x } else { cols - 1 };

            if start_x >= cols { continue; } // Skip if start_x is out of bounds for this line.

            let mut line_text = String::new();
            // Iterate from start_x up to and including end_x, capped by line length or grid width
            for x_idx in start_x..=std::cmp::min(end_x, cols - 1) {
                if x_idx < line.len() {
                    line_text.push(line[x_idx].c);
                } else {
                    // If selection goes beyond actual line length but within grid width,
                    // it's typically treated as spaces.
                    line_text.push(' ');
                }
            }

            // Handle right-trimming based on common terminal selection behavior:
            // If it's not the last line of a multi-line selection, or if the selection
            // on this line extends to the very end of the grid, we don't trim.
            // Otherwise (single line selection, or last line of multi-line, not ending at grid edge),
            // trim trailing spaces that were part of the selection rectangle but not "real" content.
            // This is a complex area with varied terminal behaviors.
            // A common behavior: trim if the selection does not extend to the next line.
            if y_idx < end_point.y { // If there's another line in the selection
                // No trim, keep trailing spaces as they are part of a continuous block
            } else {
                // This is the last (or only) line of the selection.
                // Trim trailing whitespace that might have been added due to rectangular selection
                // beyond the "logical" end of the line, unless the selection explicitly included them
                // by selecting up to the physical end of the line (cols - 1).
                if end_x < cols - 1 {
                    // If selection does not go to the end of the line, trim.
                    // This replicates behavior like xterm where selecting "foo   " results in "foo".
                    // However, if "foo   " is selected and then a newline, the spaces are kept.
                    // This is tricky. For now, a simpler approach:
                    // If it's the last line of selection, trim. If not, add \n.
                    // This might be too aggressive.
                    // Let's refine: Only trim if it's the absolute last line of the selection AND
                    // the selection didn't extend to the full width of the terminal.
                    // This is still not perfect. A common behavior is to copy the content as is from screen.
                    // The "fill with spaces" above already handles rectangular selection.
                    // The main thing is whether to add a newline.
                    // Most terminals add a newline if the selection goes to the next line,
                    // or if the selection on the last line included the "newline position" (i.e. selected the whole line).
                    // For now, the loop structure handles line by line. Add \n if not the very last char of selection.
                }
            }

            selected_text.push_str(&line_text);
            if y_idx < end_point.y {
                selected_text.push('\n');
            }
        }
        Some(selected_text)
    }

    /// Processes a string of text as if it were typed or pasted by the user.
    /// Each character is processed individually.
    /// This can trigger various terminal actions, including writing to the PTY
    /// if bracketed paste mode is not active, or internal buffering if it is.
    pub fn paste_text(&mut self, text: String) {
        // TODO: Implement bracketed paste mode.
        // If bracketed paste mode is active (e.g., `self.dec_modes.bracketed_paste_mode`),
        // the text should typically be wrapped with CSI 200 ~ and CSI 201 ~
        // and then sent to the PTY via an EmulatorAction::WritePty.
        // For now, processing char by char directly.
        // This simplistic approach might not correctly handle all pasted content,
        // especially newlines or control characters, depending on print_char's behavior.

        if self.dec_modes.bracketed_paste_mode {
            log::debug!("TerminalEmulator::paste_text - Bracketed Paste Mode ON");
            let mut pasted_bytes = Vec::new();
            pasted_bytes.extend_from_slice(b"\x1b[200~"); // Start bracketed paste
            pasted_bytes.extend_from_slice(text.as_bytes());
            pasted_bytes.extend_from_slice(b"\x1b[201~"); // End bracketed paste

            // Bracketed paste content should be sent to the PTY.
            // This requires `paste_text` to be able to return or trigger an `EmulatorAction`.
            // This is a deviation from just calling `print_char`.
            // The `input_handler` expects `paste_text` to handle this internally.
            // This implies `paste_text` might need to queue actions or have access
            // to a mechanism to send `EmulatorAction::WritePty`.
            // This is a limitation of the current design where `print_char` doesn't return actions.

            // For now, let's assume `print_char` for bracketed paste content is not the right path.
            // We need to send this as a WritePty action.
            // This cannot be done directly from here without changing signatures or
            // having the TerminalEmulator store pending actions.
            // The input_handler is the one returning EmulatorAction.
            // This suggests that the logic for bracketed paste wrapping should ideally be
            // in `input_handler.rs` when `UserInputAction::PasteText` is received.

            // Given the constraints, and that `input_handler` *calls* `paste_text`,
            // `paste_text` itself cannot easily return an `EmulatorAction::WritePty`.
            // The original `input_handler` logic for `PasteText` *did* handle bracketed paste.
            // By moving it to `TerminalEmulator::paste_text`, we've lost that direct ability.

            // Re-evaluation: The subtask says "The paste_text method in TerminalEmulator
            // will then process the string, effectively simulating keyboard input for each character.
            // This might involve pushing characters to the PTY or directly processing them as if typed.
            // For simplicity, it can call self.char_processor.process_char(ch) for each char".
            // `self.print_char()` is a better existing public method for this.

            // If bracketed paste is on, the *application* on the other side of PTY expects the wrapped sequence.
            // So, if we are "simulating keyboard input", then for bracketed paste, we should indeed
            // send the wrapped sequence to the PTY.
            // This means `paste_text` needs to return an action, or `input_handler` handles bracketed paste
            // *before* calling `paste_text`.

            // Let's revert to the `input_handler` handling bracketed paste wrapping.
            // `TerminalEmulator::paste_text` will then only handle non-bracketed paste by char processing.
            // This means the change in `input_handler.rs` was slightly off.
            // I will correct `input_handler.rs` *after* this change to `TerminalEmulator`.

            // For now, assume `paste_text` is called for non-bracketed paste, or individual chars of bracketed.
            // The subtask says "call self.char_processor.process_char(ch)". `print_char` is better.
            log::warn!("TerminalEmulator::paste_text called with bracketed paste mode ON. This mode should be handled by the caller (input_handler) by wrapping the text and sending it as WritePty. Processing char by char as fallback.");
            for ch in text.chars() {
                self.print_char(ch);
            }

        } else {
            log::debug!("TerminalEmulator::paste_text - Bracketed Paste Mode OFF. Processing {} chars.", text.len());
            for ch in text.chars() {
                self.print_char(ch); // `print_char` calls char_processor internally.
            }
        }
    }
}
