// src/term/emulator/char_processor.rs

use super::TerminalEmulator;
use crate::{
    glyph::{AttrFlags, Attributes, Glyph, WIDE_CHAR_PLACEHOLDER},
    term::{
        charset::{map_to_dec_line_drawing, CharacterSet}, // For map_char_to_active_charset
        unicode::get_char_display_width,
    },
};
use log::{trace, warn};

impl TerminalEmulator {
    /// Maps a character to its equivalent in the currently active G0/G1/G2/G3 character set.
    // This is a helper for print_char, so it can be private to this impl block.
    #[inline]
    fn map_char_to_active_charset(&self, ch: char) -> char {
        let current_set = self.active_charsets[self.active_charset_g_level];
        match current_set {
            CharacterSet::Ascii => ch,
            CharacterSet::UkNational => {
                if ch == '#' {
                    '£'
                } else {
                    ch
                }
            }
            CharacterSet::DecLineDrawing => map_to_dec_line_drawing(ch),
        }
    }

    /// Prints a single character to the terminal at the current cursor position.
    /// Handles character width, line wrapping, and updates cursor position.
    // Called from ansi_handler.rs, so pub(super) or pub.
    // pub(super) is fine as ansi_handler is a sibling module.
    pub fn print_char(&mut self, ch: char) {
        if ch == '\n' {
            self.carriage_return();
            self.move_down_one_line_and_dirty();
            return;
        }

        // Map character to the active G0/G1/G2/G3 character set.
        let ch_to_print = self.map_char_to_active_charset(ch);
        let char_width = get_char_display_width(ch_to_print);

        // Zero-width characters are complex. For now, if wcwidth reports 0, skip for cursor advancement.
        if char_width == 0 {
            trace!(
                "print_char: Encountered zero-width char '{}'. No cursor advancement.",
                ch_to_print
            );
            // TODO: Potentially handle combining characters by placing them in the current cell
            // without advancing, if the renderer and font support it.
            return;
        }

        let mut screen_ctx = self.current_screen_context();

        // Handle line wrap if cursor_wrap_next was set by the previous character.
        // This flag indicates that the cursor is at the end of the line and the next
        // character should wrap to the beginning of the next line.
        if self.cursor_wrap_next {
            self.carriage_return(); // Move to column 0 of the current line.
            self.move_down_one_line_and_dirty(); // Move to the next line, handles scrolling.
                                                 // move_down_one_line_and_dirty also resets self.cursor_wrap_next to false.
            screen_ctx = self.current_screen_context(); // Update context after potential scroll/cursor move.
                                                        // self.cursor_wrap_next is now false.
        }

        // Get current physical cursor position for placing the glyph.
        // This position is now correctly at the start of the line if a wrap just occurred.
        let (mut physical_x, mut physical_y) =
            self.cursor_controller.physical_screen_pos(&screen_ctx);

        // Check if the character (considering its width) would exceed the line width
        // from the current physical_x. This handles cases where the character is wider
        // than the remaining space on the line, even if cursor_wrap_next was false initially.
        if physical_x + char_width > screen_ctx.width {
            // If a wide char (width 2) is at the very last column (e.g. col 79 of 80), it can't fit.
            // Standard behavior: print a space in the last cell, then wrap.
            if char_width == 2 && physical_x == screen_ctx.width.saturating_sub(1) {
                let fill_glyph = Glyph {
                    c: ' ', // Fill with a space
                    attr: Attributes {
                        flags: AttrFlags::empty(),
                        ..self.cursor_controller.attributes()
                    }, // Ensure flags are clean for the space
                };
                if physical_y < self.screen.height {
                    // Bounds check
                    self.screen.set_glyph(physical_x, physical_y, fill_glyph);
                    self.screen.mark_line_dirty(physical_y);
                }
            }

            // Perform wrap: CR then effectively LF.
            self.carriage_return();
            self.move_down_one_line_and_dirty(); // This moves cursor down and handles scrolling.
                                                 // It also resets self.cursor_wrap_next.
            screen_ctx = self.current_screen_context(); // Update context
                                                        // Get new physical cursor position after this wrap.
            (physical_x, physical_y) = self.cursor_controller.physical_screen_pos(&screen_ctx);
        }

        // Place the character glyph on the screen.
        let glyph_attrs = self.cursor_controller.attributes();
        if physical_y < self.screen.height {
            // Ensure y is within bounds before writing
            let glyph_to_set = Glyph {
                c: ch_to_print,
                attr: glyph_attrs, // Use the full attributes from the cursor
            };

            self.screen.set_glyph(
                physical_x,
                physical_y,
                glyph_to_set.clone(), // Use clone if Glyph is not Copy
            );
            self.screen.mark_line_dirty(physical_y); // Mark line dirty via screen method.

            // If it's a wide character, place a placeholder and set flags.
            if char_width == 2 {
                // Mark the primary part of the wide character.
                let mut primary_glyph_attrs = glyph_attrs; // Start with cursor attributes
                primary_glyph_attrs
                    .flags
                    .insert(AttrFlags::WIDE_CHAR_PRIMARY);
                primary_glyph_attrs
                    .flags
                    .remove(AttrFlags::WIDE_CHAR_SPACER); // Ensure spacer flag is not present
                let primary_glyph = Glyph {
                    c: ch_to_print,
                    attr: primary_glyph_attrs,
                };
                self.screen.set_glyph(physical_x, physical_y, primary_glyph);

                if physical_x + 1 < screen_ctx.width {
                    let mut spacer_attrs = glyph_attrs; // Start with cursor attributes
                    spacer_attrs.flags.remove(AttrFlags::WIDE_CHAR_PRIMARY); // Ensure primary flag is not on spacer
                    spacer_attrs.flags.insert(AttrFlags::WIDE_CHAR_SPACER);
                    let placeholder_glyph = Glyph {
                        c: WIDE_CHAR_PLACEHOLDER, // Defined in glyph.rs
                        attr: spacer_attrs,
                    };
                    self.screen
                        .set_glyph(physical_x + 1, physical_y, placeholder_glyph);
                    // Line is already marked dirty from the primary character.
                } else {
                    // This case implies a wide char was printed at the exact last column.
                    // The WIDE_CHAR_PRIMARY flag is set, but no spacer is placed.
                    // The cursor advancement logic below will handle cursor_wrap_next.
                    trace!(
                        "Wide char placeholder for '{}' at ({},{}) could not be placed as it's at the edge of screen (width {}). Only primary part written.",
                        ch_to_print, physical_x, physical_y, screen_ctx.width
                    );
                }
            }
        } else {
            warn!(
                "print_char: Attempted to print at physical_y {} out of bounds (height {})",
                physical_y, self.screen.height
            );
        }

        // Advance the logical cursor position by the character's width.
        // self.cursor_controller.move_right uses the current logical position and advances it.
        // The logical position should be correct after any wrapping.
        self.cursor_controller.move_right(char_width, &screen_ctx);

        // Check if the new logical cursor position requires a wrap on the *next* character.
        let (final_logical_x, _) = self.cursor_controller.logical_pos();
        // Set cursor_wrap_next if the cursor is exactly at or beyond the width.
        // e.g., width 80 (cols 0-79). If final_logical_x is 80, it's at the wrap position.
        self.cursor_wrap_next = final_logical_x >= screen_ctx.width;
        if self.cursor_wrap_next {
            println!("cursor_wrap_next set to true. final_logical_x: {}, screen_ctx.width: {}", final_logical_x, screen_ctx.width);
        }
    }
}
