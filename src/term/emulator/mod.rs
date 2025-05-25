// src/term/emulator/mod.rs

// Crate-level imports
use crate::{
    config,
    glyph::Attributes,
    term::{
        action::EmulatorAction, // Added UserInputAction, ControlEvent
        charset::CharacterSet,
        cursor::{CursorController, ScreenContext},
        modes::{DecModeConstant, DecPrivateModes},
        screen::Screen,
        EmulatorInput, // Added EmulatorInput
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
enum FocusState {
    Focused,
    Unfocused,
}

/// The core terminal emulator.
pub struct TerminalEmulator {
    pub(super) screen: Screen,
    pub(super) focus_sate: FocusState,
    pub(super) cursor_controller: CursorController,
    pub(super) dec_modes: DecPrivateModes,
    pub(super) active_charsets: [CharacterSet; 4],
    pub(super) active_charset_g_level: usize,
    pub(super) cursor_wrap_next: bool,
    pub(super) current_cursor_shape: u16, // Stores the current cursor shape code
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
            focus_sate: FocusState::Focused,
            active_charset_g_level: 0, // Default to G0
            cursor_wrap_next: false,
            current_cursor_shape: config::CONFIG., // Use constant for default
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

    // --- Public Accessor Methods for Tests ---
    #[allow(dead_code)]
    pub(super) fn is_origin_mode_active(&self) -> bool {
        self.dec_modes.origin_mode
    }
    #[allow(dead_code)]
    pub(super) fn is_cursor_keys_app_mode_active(&self) -> bool {
        self.dec_modes.cursor_keys_app_mode
    }
    #[allow(dead_code)]
    pub(super) fn is_bracketed_paste_mode_active(&self) -> bool {
        self.dec_modes.bracketed_paste_mode
    }
    #[allow(dead_code)]
    pub(super) fn is_focus_event_mode_active(&self) -> bool {
        self.dec_modes.focus_event_mode
    }

    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub(super) fn get_cursor_shape(&self) -> u16 {
        self.current_cursor_shape
    }

    /// Returns the current logical cursor position (0-based column, row).
    pub fn cursor_pos(&self) -> (usize, usize) {
        self.cursor_controller.logical_pos()
    }

    /// Returns `true` if the alternate screen buffer is currently active.
    pub fn is_alt_screen_active(&self) -> bool {
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
}
