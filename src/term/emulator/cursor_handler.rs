// src/term/emulator/cursor_handler.rs

use super::TerminalEmulator;
use crate::term::action::EmulatorAction; // For dimensions
use log::warn; // For WindowManipulation

impl TerminalEmulator {
    pub(super) fn backspace(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_left(1);
    }

    pub(super) fn horizontal_tab(&mut self) {
        self.cursor_wrap_next = false;
        let (current_x, _) = self.cursor_controller.logical_pos();
        let screen_ctx = self.current_screen_context();
        let next_stop = self
            .screen
            .get_next_tabstop(current_x)
            .unwrap_or(screen_ctx.width.saturating_sub(1).max(current_x));
        self.cursor_controller
            .move_to_logical_col(next_stop, &screen_ctx);
    }

    pub(super) fn cursor_up(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_up(n);
    }

    pub(super) fn cursor_down(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        // The original cursor_down in ansi_handler called index().
        // Here, we assume index() is a method on TerminalEmulator or will be handled by the caller.
        // For now, let's replicate the direct effect or call index if it's available.
        // This might need adjustment based on where index() is finally located.
        // For now, let's assume `index` is a method on `TerminalEmulator`
        // that can be called from here if needed, or this method's body will be
        // adjusted based on how `ansi_handler` calls it.
        // The subtask is to move the CSI helpers. The original `cursor_down` in ansi_handler
        // called `index(emulator)`. If `index` is a method on `TerminalEmulator`, this becomes `self.index()`.
        log::trace!("cursor_handler::cursor_down: n = {}", n);
        for _i in 0..n {
            self.index(); // Assuming index is a pub(super) method in methods.rs or similar
        }
    }

    pub(super) fn cursor_forward(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller
            .move_right(n, &self.current_screen_context());
    }

    pub(super) fn cursor_backward(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_left(n);
    }

    pub(super) fn cursor_to_column(&mut self, col: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller
            .move_to_logical_col(col, &self.current_screen_context());
    }

    pub(super) fn cursor_to_pos(&mut self, row_param: usize, col_param: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_to_logical(
            col_param,
            row_param,
            &self.current_screen_context(),
        );
    }

    // This method handles CsiCommand::WindowManipulation
    pub(super) fn handle_window_manipulation(
        &mut self,
        ps1: u16,
        _ps2: Option<u16>, // ps2 and ps3 are not used in the current implementation
        _ps3: Option<u16>,
    ) -> Option<EmulatorAction> {
        match ps1 {
            14 => {
                warn!(
                    "WindowManipulation: Report text area size in pixels (14) requested, but not implemented."
                );
                None
            }
            18 => {
                // Assuming dimensions() is available via TerminalInterface trait
                let (cols, rows) = self.dimensions();
                let response = format!("\x1b[8;{};{}t", rows, cols);
                Some(EmulatorAction::WritePty(response.into_bytes()))
            }
            22 | 23 => {
                warn!("WindowManipulation: Save/Restore window title (22/23) not implemented.");
                None
            }
            _ => {
                warn!(
                    "Unhandled WindowManipulation: ps1={}, ps2={:?}, ps3={:?}",
                    ps1, _ps2, _ps3
                );
                None
            }
        }
    }
}
