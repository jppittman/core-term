// src/term/emulator/cursor_handler.rs

use super::TerminalEmulator;
use crate::term::action::EmulatorAction;
use log::warn;

const XTWINOPS_RESIZE_TEXT_AREA: u16 = 8;
const XTWINOPS_REPORT_TEXT_AREA_SIZE_PIXELS: u16 = 14;
const XTWINOPS_REPORT_TEXT_AREA_SIZE_CHARS: u16 = 18;
const XTWINOPS_SAVE_TITLE: u16 = 22;
const XTWINOPS_RESTORE_TITLE: u16 = 23;

impl TerminalEmulator {
    /// Moves the cursor one position to the left.
    ///
    /// This corresponds to the Backspace (BS) C0 control character.
    /// If the cursor is already at the left margin, it stays there.
    /// Resets the `cursor_wrap_next` flag.
    pub(super) fn backspace(&mut self) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_left(1);
    }

    /// Moves the cursor to the next tab stop.
    ///
    /// This corresponds to the Horizontal Tab (HT) C0 control character.
    /// If there are no more tab stops, it moves to the right margin.
    /// Resets the `cursor_wrap_next` flag.
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

    /// Moves the cursor up by `n` rows.
    ///
    /// This corresponds to the CUU (Cursor Up) ANSI sequence.
    /// The cursor stops at the top margin (or top of scrolling region).
    /// Resets the `cursor_wrap_next` flag.
    pub(super) fn cursor_up(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_up(n);
    }

    /// Moves the cursor down by `n` rows.
    ///
    /// This corresponds to the CUD (Cursor Down) ANSI sequence.
    ///
    /// Note: This implementation currently calls `index()` `n` times, which means
    /// it *will* scroll the screen if the cursor is at the bottom margin.
    /// This is a known deviation from the standard CUD behavior (which should clamp).
    pub(super) fn cursor_down(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        log::trace!("cursor_handler::cursor_down: n = {}", n);
        for _i in 0..n {
            self.index();
        }
    }

    /// Moves the cursor forward (right) by `n` columns.
    ///
    /// This corresponds to the CUF (Cursor Forward) ANSI sequence.
    /// The cursor stops at the right margin.
    /// Resets the `cursor_wrap_next` flag.
    pub(super) fn cursor_forward(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller
            .move_right(n, &self.current_screen_context());
    }

    /// Moves the cursor backward (left) by `n` columns.
    ///
    /// This corresponds to the CUB (Cursor Backward) ANSI sequence.
    /// The cursor stops at the left margin.
    /// Resets the `cursor_wrap_next` flag.
    pub(super) fn cursor_backward(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_left(n);
    }

    /// Moves the cursor to a specific absolute column (0-based index).
    ///
    /// This corresponds to the CHA (Cursor Horizontal Absolute) ANSI sequence.
    /// Resets the `cursor_wrap_next` flag.
    pub(super) fn cursor_to_column(&mut self, col: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller
            .move_to_logical_col(col, &self.current_screen_context());
    }

    /// Moves the cursor to a specific absolute position (row, col).
    ///
    /// This corresponds to the CUP (Cursor Position) ANSI sequence.
    /// Coordinates are 0-based logical coordinates.
    /// Resets the `cursor_wrap_next` flag.
    pub(super) fn cursor_to_pos(&mut self, row_param: usize, col_param: usize) {
        self.cursor_wrap_next = false;
        self.cursor_controller.move_to_logical(
            col_param,
            row_param,
            &self.current_screen_context(),
        );
    }

    /// Handles XTWINOPS window manipulation commands (CSI 18;...t).
    ///
    /// Supports:
    /// - Resize text area (8)
    /// - Report text area size in characters (18)
    ///
    /// Returns an `EmulatorAction` if the command requires external handling (e.g., resizing PTY).
    pub(super) fn handle_window_manipulation(
        &mut self,
        ps1: u16,
        ps2: Option<u16>,
        ps3: Option<u16>,
    ) -> Option<EmulatorAction> {
        match ps1 {
            XTWINOPS_RESIZE_TEXT_AREA => {
                let rows = ps2?;
                let cols = ps3?;

                if rows == 0 || cols == 0 {
                    return None;
                }

                self.resize(cols as usize, rows as usize);
                Some(EmulatorAction::ResizePty { cols, rows })
            }
            XTWINOPS_REPORT_TEXT_AREA_SIZE_PIXELS => {
                warn!("WindowManipulation: Report text area size in pixels (14) not implemented");
                None
            }
            XTWINOPS_REPORT_TEXT_AREA_SIZE_CHARS => {
                let (cols, rows) = self.dimensions();
                let response = format!("\x1b[8;{};{}t", rows, cols);
                Some(EmulatorAction::WritePty(response.into_bytes()))
            }
            XTWINOPS_SAVE_TITLE | XTWINOPS_RESTORE_TITLE => {
                warn!("WindowManipulation: Save/Restore window title (22/23) not implemented");
                None
            }
            _ => {
                warn!(
                    "Unhandled WindowManipulation: ps1={}, ps2={:?}, ps3={:?}",
                    ps1, ps2, ps3
                );
                None
            }
        }
    }
}
