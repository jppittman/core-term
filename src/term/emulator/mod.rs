// src/term/emulator/mod.rs

// Crate-level imports
use crate::{
    glyph::{Attributes, Glyph}, // Ensure Glyph and Attributes are imported
    term::{
        EmulatorInput,          // Added EmulatorInput
        action::EmulatorAction, // Added UserInputAction, ControlEvent
        charset::CharacterSet,
        cursor::{self, CursorController, ScreenContext}, // Import cursor module for its CursorShape
        modes::{DecModeConstant, DecPrivateModes},
        screen::Screen,
        snapshot::{
            CursorRenderState, CursorShape, RenderSnapshot, SelectionMode, SelectionRenderState,
            SnapshotLine,
        },
    },
};

// Logging (optional, but good practice if used)
use log::{debug, warn};

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

    pub(super) fn is_origin_mode_active(&self) -> bool {
        self.dec_modes.origin_mode
    }

    pub(super) fn is_cursor_keys_app_mode_active(&self) -> bool {
        self.dec_modes.cursor_keys_app_mode
    }

    pub(super) fn is_bracketed_paste_mode_active(&self) -> bool {
        self.dec_modes.bracketed_paste_mode
    }

    pub(super) fn is_focus_event_mode_active(&self) -> bool {
        self.dec_modes.focus_event_mode
    }

    pub(super) fn is_mouse_mode_active(&self, mode_num: u16) -> bool {
        match DecModeConstant::from_u16(mode_num) {
            Some(DecModeConstant::MouseX10) => self.dec_modes.mouse_x10_mode,
            Some(DecModeConstant::MouseVt200) => self.dec_modes.mouse_vt200_mode,
            Some(DecModeConstant::MouseVt200Highlight) => self.dec_modes.mouse_vt200_highlight_mode,
            Some(DecModeConstant::MouseButtonEvent) => self.dec_modes.mouse_button_event_mode,
            Some(DecModeConstant::MouseAnyEvent) => self.dec_modes.mouse_any_event_mode,
            Some(DecModeConstant::MouseUtf8) => self.dec_modes.mouse_utf8_mode,
            Some(DecModeConstant::MouseSgr) => self.dec_modes.mouse_sgr_mode,
            Some(DecModeConstant::MouseUrxvt) => {
                warn!("is_mouse_mode_active check for MouseUrxvt (1015): Not fully implemented.");
                false
            }
            Some(DecModeConstant::MousePixelPosition) => {
                warn!(
                    "is_mouse_mode_active check for MousePixelPosition (1016): Not fully implemented."
                );
                false
            }
            _ => {
                warn!(
                    "is_mouse_mode_active called with non-mouse mode or unhandled mouse mode: {}",
                    mode_num
                );
                false
            }
        }
    }

    /// Returns the current logical cursor position (0-based column, row).
    pub(super) fn cursor_pos(&self) -> (usize, usize) {
        self.cursor_controller.logical_pos()
    }

    /// Returns `true` if the alternate screen buffer is currently active.
    pub(super) fn is_alt_screen_active(&self) -> bool {
        self.screen.alt_screen_active
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

        // Selection state is None for now as per plan.
        let selection_state = None; // TODO: Populate this when selection is handled

        RenderSnapshot {
            dimensions: (width, height),
            lines,
            cursor_state,
            selection_state,
        }
    }
}
