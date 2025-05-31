// src/term/emulator/methods.rs

// Note: This file is part of the `emulator` module.
// `TerminalEmulator` struct is defined in `src/term/emulator/mod.rs`
use super::TerminalEmulator; // Bring the struct into scope from the parent module

// Crate-level imports for items outside src/term/
use crate::glyph::Attributes;

impl TerminalEmulator {
    pub(super) fn carriage_return(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller.carriage_return();
    }

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
}
