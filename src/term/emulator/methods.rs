// src/term/emulator/methods.rs

// Note: This file is part of the `emulator` module.
// `TerminalEmulator` struct is defined in `src/term/emulator/mod.rs`
use super::TerminalEmulator; // Bring the struct into scope from the parent module
use super::ansi_handler; // Use the new ansi_handler module
use super::input_handler; // Use the new input_handler module
// For the `dimensions()` method

// Corrected Crate-level imports for items within src/term/
use crate::term::{ControlEvent, action::EmulatorAction};

// Standard library imports

// Crate-level imports for items outside src/term/
use crate::{ansi::commands::AnsiCommand, glyph::Attributes};

// Logging

// Constants (ensure these are defined, e.g., in term/mod.rs or config.rs if not already)
const DEFAULT_CURSOR_SHAPE: u16 = 1; // Example default shape
const DEFAULT_TAB_INTERVAL: u8 = 8;

// Note: The TerminalEmulator struct and several of its methods have been moved to src/term/emulator/mod.rs
// The methods remaining in this file are part of the `impl TerminalEmulator` block.

impl TerminalEmulator {
    /// Handles a parsed `AnsiCommand`.
    pub(super) fn handle_ansi_command(&mut self, command: AnsiCommand) -> Option<EmulatorAction> {
        // Delegate to the new handler in ansi_handler.rs
        ansi_handler::process_ansi_command(self, command)
    }

    // Helper methods like backspace, horizontal_tab, perform_line_feed, carriage_return,
    // set_g_level, index, reverse_index, save_cursor_dec, restore_cursor_dec,
    // designate_character_set, and the small CSI cursor movement helpers
    // have been moved to ansi_handler.rs and are now private functions there.

    // Larger methods that were called by handle_ansi_command (like erase_in_display,
    // handle_sgr_attributes, etc.) remain here as methods on TerminalEmulator.
    // They will be called via `emulator.method_name()` from ansi_handler.rs.

    /// Handles a `BackendEvent`.
    pub(super) fn handle_user_event(
        &mut self,
        event: crate::term::UserInputAction,
    ) -> Option<EmulatorAction> {
        input_handler::process_user_input_action(self, event)
    }

    /// Handles an internal `ControlEvent`.
    pub(super) fn handle_control_event(&mut self, event: ControlEvent) -> Option<EmulatorAction> {
        input_handler::process_control_event(self, event)
    }

    // The following methods (`resize`, `cursor_pos`, `is_alt_screen_active`) were moved to src/term/emulator/mod.rs
    // and are removed from here to avoid duplicate definitions:
    // - resize
    // - cursor_pos
    // - is_alt_screen_active

    // --- Character Printing and Low-Level Operations ---
    // `print_char` and `map_char_to_active_charset` have been moved to char_processor.rs

    // Note: `backspace`, `horizontal_tab`, `perform_line_feed`, (now in ansi_handler or methods.rs)
    // `carriage_return`, `set_g_level`, `designate_character_set`, `index`, `reverse_index`,
    // `save_cursor_dec`, `restore_cursor_dec` and CSI sub-handlers like `cursor_up`, `cursor_down`,
    // `cursor_forward`, `cursor_backward`, `cursor_to_column`, `cursor_to_pos`
    // were moved to ansi_handler.rs

    // Methods like erase_in_display, erase_in_line etc. are kept here as they are larger
    // and might be refactored into their own modules later.
    // They are called from ansi_handler.rs via `emulator.method_name()`.

    // --- Methods moved back from ansi_handler or kept for use by print_char/handle_set_mode ---
    pub(super) fn carriage_return(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller.carriage_return();
    }

    // move_down_one_line_and_dirty has been moved to screen_ops.rs
    // erase_in_display has been moved to screen_ops.rs
    // erase_in_line has been moved to screen_ops.rs
    // erase_chars has been moved to screen_ops.rs
    // insert_blank_chars has been moved to screen_ops.rs
    // delete_chars has been moved to screen_ops.rs
    // insert_lines has been moved to screen_ops.rs
    // delete_lines has been moved to screen_ops.rs
    // scroll_up has been moved to screen_ops.rs
    // scroll_down has been moved to screen_ops.rs

    pub(super) fn save_cursor_dec(&mut self) {
        self.cursor_controller.save_state();
    }

    pub(super) fn restore_cursor_dec(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller
            .restore_state(&self.current_screen_context(), Attributes::default());
        self.screen.default_attributes = self.cursor_controller.attributes();
    }

    pub(super) fn index(&mut self) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, current_physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);

        if current_physical_y == screen_ctx.scroll_bot {
            self.screen.scroll_up_serial(1);
        } else if current_physical_y < screen_ctx.height.saturating_sub(1) {
            self.cursor_controller.move_down(1, &screen_ctx);
        }
        if current_physical_y < self.screen.height {
            self.screen.mark_line_dirty(current_physical_y);
        }
        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context());
        if current_physical_y != new_physical_y && new_physical_y < self.screen.height {
            self.screen.mark_line_dirty(new_physical_y);
        }
    }
    // --- End of methods moved back or to be kept ---

    // Methods like handle_sgr_attributes, handle_set_mode, handle_window_manipulation, handle_osc
    // remain here as they are more about mode setting and attribute handling rather than direct screen ops.
    // `handle_sgr_attributes` and `handle_set_mode` have been moved to mode_handler.rs
    // `handle_osc` has been moved to osc_handler.rs
    // `handle_window_manipulation` has been moved to cursor_handler.rs
}
