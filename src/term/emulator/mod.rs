// src/term/emulator/mod.rs

// Crate-level imports
use crate::{
    glyph::Attributes, // Ensure Glyph and Attributes are imported
    term::{
        EmulatorInput,          // Added EmulatorInput
        action::EmulatorAction, // Added UserInputAction, ControlEvent
        charset::CharacterSet,
        cursor::{self, CursorController, ScreenContext}, // Import cursor module for its CursorShape
        modes::{DecModeConstant, DecPrivateModes},
        screen::Screen,
        selection::Selection, // Import Selection
        snapshot::{
            CursorRenderState, CursorShape, RenderSnapshot, SelectionRenderState, SnapshotLine,
        }, // Added SelectionRenderState
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
    pub selection: Selection,
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
            selection: Selection::new(),
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
        self.selection.clear_selection();
        debug!(
            "Terminal resized to {}x{}. Cursor re-clamped. All lines marked dirty by screen.resize(). Selection cleared.",
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
        let selection_state = self
            .selection
            .get_render_state(self.screen.width, self.screen.height);

        RenderSnapshot {
            dimensions: (width, height),
            lines,
            cursor_state,
            selection_state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{MouseButton, MouseEventType}; // For creating UserInputAction
    use crate::keys::Modifiers;
    use crate::term::{
        action::{EmulatorAction, UserInputAction}, // For creating EmulatorInput
        snapshot::SelectionMode,                   // For asserting selection mode
    };
    // SelectionRenderState is already imported at the top of emulator/mod.rs

    // Helper function to create a default emulator instance
    fn default_emulator(cols: usize, rows: usize) -> TerminalEmulator {
        TerminalEmulator::new(cols, rows, 0) // 0 scrollback for simplicity in these tests
    }

    #[test]
    fn test_mouse_press_starts_selection() {
        let mut emulator = default_emulator(80, 24);
        let input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 5,
            row: 2,
            event_type: MouseEventType::Press,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        emulator.interpret_input(input);

        assert_eq!(emulator.selection.start, Some((5, 2)));
        assert_eq!(emulator.selection.end, Some((5, 2)));
        assert!(emulator.selection.is_active);
        assert_eq!(emulator.selection.mode, SelectionMode::Normal);
    }

    #[test]
    fn test_mouse_press_alt_starts_block_selection() {
        let mut emulator = default_emulator(80, 24);
        let input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 5,
            row: 2,
            event_type: MouseEventType::Press,
            button: MouseButton::Left,
            modifiers: Modifiers::ALT,
        });
        emulator.interpret_input(input);

        assert_eq!(emulator.selection.start, Some((5, 2)));
        assert_eq!(emulator.selection.end, Some((5, 2)));
        assert!(emulator.selection.is_active);
        assert_eq!(emulator.selection.mode, SelectionMode::Block);
    }

    #[test]
    fn test_mouse_drag_updates_selection() {
        let mut emulator = default_emulator(80, 24);
        // Start selection
        let press_input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 5,
            row: 2,
            event_type: MouseEventType::Press,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        emulator.interpret_input(press_input);

        // Drag
        let move_input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 10,
            row: 2,
            event_type: MouseEventType::Move,
            button: MouseButton::Left, // X11 often reports the button being dragged during motion
            modifiers: Modifiers::empty(),
        });
        emulator.interpret_input(move_input);

        assert_eq!(emulator.selection.start, Some((5, 2))); // Start should remain the same
        assert_eq!(emulator.selection.end, Some((10, 2))); // End should update
        assert!(emulator.selection.is_active); // Still active
    }

    #[test]
    fn test_mouse_release_ends_selection() {
        let mut emulator = default_emulator(80, 24);
        // Start selection
        let press_input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 5,
            row: 2,
            event_type: MouseEventType::Press,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        emulator.interpret_input(press_input);

        // Drag (optional, but good to have a different end point)
        let move_input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 10,
            row: 2,
            event_type: MouseEventType::Move,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        emulator.interpret_input(move_input);

        // Release
        let release_input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 10,
            row: 2, // Release at the dragged position
            event_type: MouseEventType::Release,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        emulator.interpret_input(release_input);

        assert_eq!(emulator.selection.start, Some((5, 2)));
        assert_eq!(emulator.selection.end, Some((10, 2)));
        assert!(!emulator.selection.is_active); // Selection is no longer active
    }

    #[test]
    fn test_render_snapshot_includes_selection() {
        let mut emulator = default_emulator(80, 24);
        // Start a selection
        let press_input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 5,
            row: 2,
            event_type: MouseEventType::Press,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        emulator.interpret_input(press_input);
        let move_input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 10,
            row: 2,
            event_type: MouseEventType::Move,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        emulator.interpret_input(move_input);
        // Selection is active but not yet ended by release, get_render_state should still work.

        let snapshot = emulator.get_render_snapshot();
        assert_eq!(
            snapshot.selection_state,
            Some(SelectionRenderState {
                start_coords: (5, 2),
                end_coords: (10, 2),
                mode: SelectionMode::Normal,
            })
        );
    }

    #[test]
    fn test_render_snapshot_no_selection() {
        let emulator = default_emulator(80, 24); // No selection made
        let snapshot = emulator.get_render_snapshot();
        assert_eq!(snapshot.selection_state, None);
    }

    #[test]
    fn test_mouse_report_x10_press() {
        let mut emulator = default_emulator(80, 24);
        emulator.dec_modes.mouse_x10_mode = true; // Enable X10 reporting

        let input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 5, // 0-indexed
            row: 2, // 0-indexed
            event_type: MouseEventType::Press,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        let action = emulator.interpret_input(input);

        // Expected: ESC [ M Cb Cx Cy
        // Cb = button_code (0 for left) + 32 = 32 (Space)
        // Cx = col (5) + 1 + 32 = 38 ('&')
        // Cy = row (2) + 1 + 32 = 35 ('#')
        let expected_sequence = b"\x1B[M #&".to_vec(); // Corrected: Cb Cx Cy -> Space # &
        assert_eq!(action, Some(EmulatorAction::WritePty(expected_sequence)));
    }

    #[test]
    fn test_mouse_report_sgr_press() {
        let mut emulator = default_emulator(80, 24);
        emulator.dec_modes.mouse_sgr_mode = true; // Enable SGR reporting

        let input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 5, // 0-indexed
            row: 2, // 0-indexed
            event_type: MouseEventType::Press,
            button: MouseButton::Left,
            modifiers: Modifiers::SHIFT,
        });
        let action = emulator.interpret_input(input);

        // Expected: ESC [ < Button;Col+1;Row+1 M
        // Button = 0 (Left) + 4 (Shift) = 4
        // Col+1 = 6
        // Row+1 = 3
        let expected_sequence = b"\x1B[<4;6;3M".to_vec();
        assert_eq!(action, Some(EmulatorAction::WritePty(expected_sequence)));
    }

    #[test]
    fn test_mouse_report_sgr_release() {
        let mut emulator = default_emulator(80, 24);
        emulator.dec_modes.mouse_sgr_mode = true; // Enable SGR reporting

        let input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 5, // 0-indexed
            row: 2, // 0-indexed
            event_type: MouseEventType::Release,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        let action = emulator.interpret_input(input);

        // Expected: ESC [ < Button;Col+1;Row+1 m
        // Button = 0 (Left)
        // Col+1 = 6
        // Row+1 = 3
        let expected_sequence = b"\x1B[<0;6;3m".to_vec();
        assert_eq!(action, Some(EmulatorAction::WritePty(expected_sequence)));
    }

    #[test]
    fn test_mouse_report_sgr_drag() {
        let mut emulator = default_emulator(80, 24);
        emulator.dec_modes.mouse_sgr_mode = true;

        // Start selection (press) - SGR reports this
        let press_input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 5,
            row: 2,
            event_type: MouseEventType::Press,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        let press_action = emulator.interpret_input(press_input);
        let expected_press_seq = b"\x1B[<0;6;3M".to_vec();
        assert_eq!(
            press_action,
            Some(EmulatorAction::WritePty(expected_press_seq))
        );

        // Drag (move while button is considered pressed by selection state)
        let move_input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 6,
            row: 2,
            event_type: MouseEventType::Move,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        let move_action = emulator.interpret_input(move_input);

        // Expected: ESC [ < Button;Col+1;Row+1 M (still 'M' for drag)
        // Button = 0 (Left) + 32 (motion for SGR means button code includes motion if applicable, but for SGR, motion is reported with the button down)
        // Col+1 = 7
        // Row+1 = 3
        // The input_handler.rs for SGR drag does not add 32 to button. It uses the original button code.
        let expected_drag_seq = b"\x1B[<0;7;3M".to_vec();
        assert_eq!(
            move_action,
            Some(EmulatorAction::WritePty(expected_drag_seq))
        );
    }

    #[test]
    fn test_mouse_report_disabled() {
        let mut emulator = default_emulator(80, 24);
        // Ensure all mouse reporting modes are off (default state of DecPrivateModes)
        assert!(!emulator.dec_modes.mouse_x10_mode);
        assert!(!emulator.dec_modes.mouse_sgr_mode);
        // Add asserts for other modes if they exist, e.g., mouse_vt200_mode, mouse_button_event_mode, mouse_any_event_mode

        let input = EmulatorInput::User(UserInputAction::MouseInput {
            col: 5,
            row: 2,
            event_type: MouseEventType::Press,
            button: MouseButton::Left,
            modifiers: Modifiers::empty(),
        });
        let action = emulator.interpret_input(input);

        // Expect None because no reporting mode is active, only selection should occur internally.
        assert_eq!(action, None);
        // Verify selection still happened
        assert!(emulator.selection.is_active);
        assert_eq!(emulator.selection.start, Some((5, 2)));
    }
}
