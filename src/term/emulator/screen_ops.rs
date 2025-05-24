// src/term/emulator/screen_ops.rs

use super::TerminalEmulator;
use crate::glyph::Attributes; // For default_attributes
use crate::term::modes::EraseMode; // EraseMode is used by erase_in_display, erase_in_line
use std::cmp::min; // For erase_chars

use log::{trace, warn}; // For move_down_one_line_and_dirty and erase_in_display/erase_in_line

impl TerminalEmulator {
    pub(super) fn erase_in_display(&mut self, mode: EraseMode) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        match mode {
            EraseMode::ToEnd => {
                self.screen
                    .clear_line_segment(cy_phys, cx_phys, screen_ctx.width);
                for y in (cy_phys + 1)..screen_ctx.height {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
            }
            EraseMode::ToStart => {
                for y in 0..cy_phys {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
                self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1);
            }
            EraseMode::All => {
                for y in 0..screen_ctx.height {
                    self.screen.clear_line_segment(y, 0, screen_ctx.width);
                }
            }
            EraseMode::Scrollback => {
                self.screen.scrollback.clear();
                return;
            }
            EraseMode::Unknown => warn!("Unknown ED mode used."),
        }
        if mode != EraseMode::Scrollback {
            self.screen.mark_all_dirty();
        }
    }

    pub(super) fn erase_in_line(&mut self, mode: EraseMode) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        match mode {
            EraseMode::ToEnd => self
                .screen
                .clear_line_segment(cy_phys, cx_phys, screen_ctx.width),
            EraseMode::ToStart => self.screen.clear_line_segment(cy_phys, 0, cx_phys + 1),
            EraseMode::All => self.screen.clear_line_segment(cy_phys, 0, screen_ctx.width),
            EraseMode::Scrollback => {
                warn!("EraseMode::Scrollback is not applicable to EraseInLine (EL).")
            }
            EraseMode::Unknown => warn!("Unknown EL mode used."),
        }
    }

    pub(super) fn erase_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_phys, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        let end_x = min(cx_phys + n, screen_ctx.width);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.clear_line_segment(cy_phys, cx_phys, end_x);
    }

    pub(super) fn insert_blank_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.insert_blank_chars_in_line(cy_phys, cx_log, n);
    }

    pub(super) fn delete_chars(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (cx_log, _) = self.cursor_controller.logical_pos();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.delete_chars_in_line(cy_phys, cx_log, n);
    }

    pub(super) fn insert_lines(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        if cy_phys >= screen_ctx.scroll_top && cy_phys <= screen_ctx.scroll_bot {
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();

            self.screen
                .set_scrolling_region(cy_phys + 1, original_scroll_bottom + 1);
            self.screen.scroll_down_serial(n);

            self.screen
                .set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);

            for y_dirty in cy_phys..=original_scroll_bottom {
                if y_dirty < self.screen.height {
                    self.screen.mark_line_dirty(y_dirty);
                }
            }
        }
    }

    pub(super) fn delete_lines(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        let screen_ctx = self.current_screen_context();
        let (_, cy_phys) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        self.screen.default_attributes = self.cursor_controller.attributes();

        if cy_phys >= screen_ctx.scroll_top && cy_phys <= screen_ctx.scroll_bot {
            let original_scroll_top = self.screen.scroll_top();
            let original_scroll_bottom = self.screen.scroll_bot();

            self.screen
                .set_scrolling_region(cy_phys + 1, original_scroll_bottom + 1);
            self.screen.scroll_up_serial(n);

            self.screen
                .set_scrolling_region(original_scroll_top + 1, original_scroll_bottom + 1);
            for y_dirty in cy_phys..=original_scroll_bottom {
                if y_dirty < self.screen.height {
                    self.screen.mark_line_dirty(y_dirty);
                }
            }
        }
    }

    pub(super) fn scroll_up(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.scroll_up_serial(n);
    }

    pub(super) fn scroll_down(&mut self, n: usize) {
        self.cursor_wrap_next = false;
        self.screen.default_attributes = self.cursor_controller.attributes();
        self.screen.scroll_down_serial(n);
    }

    pub(super) fn move_down_one_line_and_dirty(&mut self) {
        self.cursor_wrap_next = false; // Critical: any vertical movement resets pending wrap.
        let screen_ctx = self.current_screen_context();
        let (_, current_logical_y) = self.cursor_controller.logical_pos();
        let (_current_physical_x, current_physical_y) =
            self.cursor_controller.physical_screen_pos(&screen_ctx);

        let max_logical_y_in_region = if screen_ctx.origin_mode_active {
            screen_ctx.scroll_bot.saturating_sub(screen_ctx.scroll_top)
        } else {
            screen_ctx.height.saturating_sub(1)
        };

        let physical_effective_bottom = if screen_ctx.origin_mode_active {
            screen_ctx.scroll_bot
        } else {
            screen_ctx.height.saturating_sub(1)
        };

        if current_physical_y == physical_effective_bottom {
            trace!(
                "move_down_one_line: Scrolling up. Cursor at physical_y: {}, effective_bottom: {}",
                current_physical_y,
                physical_effective_bottom
            );
            self.screen.scroll_up_serial(1);
        } else if current_logical_y < max_logical_y_in_region {
            trace!(
                "move_down_one_line: Moving cursor down. logical_y: {}, max_logical_y_in_region: {}",
                current_logical_y,
                max_logical_y_in_region
            );
            self.cursor_controller.move_down(1, &screen_ctx);
        } else if !screen_ctx.origin_mode_active
            && current_physical_y < screen_ctx.height.saturating_sub(1)
        {
            trace!(
                "move_down_one_line: Moving cursor down (below scroll region, origin mode off). physical_y: {}, screen_height: {}",
                current_physical_y,
                screen_ctx.height
            );
            self.cursor_controller.move_down(1, &screen_ctx);
        } else {
            trace!(
                "move_down_one_line: Cursor at bottom, no scroll or move_down. physical_y: {}, logical_y: {}, max_logical_y: {}",
                current_physical_y,
                current_logical_y,
                max_logical_y_in_region
            );
        }

        trace!(
            "move_down_one_line: Marking old line dirty. current_physical_y: {}, screen_height: {}",
            current_physical_y,
            self.screen.height
        );
        if current_physical_y < self.screen.height {
            // Bounds check before marking dirty
            self.screen.mark_line_dirty(current_physical_y);
        }

        let (_, new_physical_y) = self
            .cursor_controller
            .physical_screen_pos(&self.current_screen_context()); // Re-fetch context as it might have changed

        if current_physical_y != new_physical_y {
            trace!(
                "move_down_one_line: Marking new line dirty. new_physical_y: {}, screen_height: {}",
                new_physical_y,
                self.screen.height
            );
            if new_physical_y < self.screen.height {
                // Bounds check
                self.screen.mark_line_dirty(new_physical_y);
            }
        } else {
            trace!(
                "move_down_one_line: New physical y ({}) is same as current ({}), not marking new line again.",
                new_physical_y,
                current_physical_y
            );
        }
    }
}
