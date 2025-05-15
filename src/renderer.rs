// src/renderer.rs

//! This module defines the `Renderer`.
//!
//! The `Renderer`'s primary responsibility is to translate the visual state of the
//! `TerminalEmulator` into a series of abstract drawing commands that can be
//! executed by a `Driver`. It is designed to be backend-agnostic.
//! It defines default foreground and background colors for resolving
//! `Color::Default` from glyph attributes.

use crate::backends::{CellCoords, CellRect, Driver, TextRunStyle};
use crate::glyph::{AttrFlags, Attributes, Color, Glyph, NamedColor};
use crate::term::TerminalInterface;
use crate::term::unicode::get_char_display_width;

use anyhow::Result;
use log::{trace, warn};
use std::collections::HashSet;

/// Default foreground color used by the renderer when a glyph specifies `Color::Default`.
const RENDERER_DEFAULT_FG: Color = Color::Named(NamedColor::White);
/// Default background color used by the renderer when a glyph specifies `Color::Default`.
const RENDERER_DEFAULT_BG: Color = Color::Named(NamedColor::Black);

/// The `Renderer` translates `TerminalEmulator` state into abstract drawing commands.
pub struct Renderer {
    // Currently stateless, but could hold caching, theming, or config options.
}

impl Renderer {
    /// Creates a new `Renderer` instance.
    pub fn new() -> Self {
        Self {}
    }

    /// Resolves `Color::Default` and handles the `AttrFlags::REVERSE` flag
    /// to determine the effective foreground, background, and flags for rendering.
    ///
    /// # Arguments
    /// * `cell_fg` - The foreground color specified by the glyph's attributes.
    /// * `cell_bg` - The background color specified by the glyph's attributes.
    /// * `cell_flags` - The attribute flags specified by the glyph.
    ///
    /// # Returns
    /// A tuple `(effective_fg, effective_bg, effective_flags)`.
    /// `effective_flags` will have `AttrFlags::REVERSE` removed if it was applied.
    fn get_effective_colors_and_flags(
        &self,
        cell_fg: Color,
        cell_bg: Color,
        cell_flags: AttrFlags,
    ) -> (Color, Color, AttrFlags) {
        let mut resolved_fg = match cell_fg {
            Color::Default => RENDERER_DEFAULT_FG,
            c => c,
        };
        let mut resolved_bg = match cell_bg {
            Color::Default => RENDERER_DEFAULT_BG,
            c => c,
        };

        let mut effective_flags = cell_flags;

        if cell_flags.contains(AttrFlags::REVERSE) {
            std::mem::swap(&mut resolved_fg, &mut resolved_bg);
            effective_flags.remove(AttrFlags::REVERSE); // REVERSE is consumed by swapping
        }
        (resolved_fg, resolved_bg, effective_flags)
    }

    /// Draws the current state of the `TerminalEmulator` using the provided `Driver`.
    ///
    /// This method orchestrates the drawing process by:
    /// 1. Determining which lines need redrawing based on dirty flags from the terminal
    ///    and the cursor's position.
    /// 2. Performing a full screen clear if a full refresh is detected (e.g., initial draw, resize).
    /// 3. Calling helper methods to draw the content of each necessary line.
    /// 4. Drawing the cursor if it's visible.
    /// 5. Presenting the completed frame to the display via the driver.
    pub fn draw(&self, term: &mut impl TerminalInterface, driver: &mut dyn Driver) -> Result<()> {
        let (term_width, term_height) = term.dimensions();

        if term_width == 0 || term_height == 0 {
            trace!(
                "Renderer::draw: Terminal dimensions zero ({}x{}), skipping draw.",
                term_width, term_height
            );
            return Ok(());
        }

        let initially_dirty_lines_from_term: HashSet<usize> =
            term.take_dirty_lines().into_iter().collect();
        let mut something_was_drawn = !initially_dirty_lines_from_term.is_empty();
        trace!(
            "Renderer::draw: Initially dirty lines from term: {:?}, term_dims: {}x{}",
            initially_dirty_lines_from_term, term_width, term_height
        );

        let is_full_refresh_heuristic = term_height > 0
            && initially_dirty_lines_from_term.len() == term_height
            && initially_dirty_lines_from_term
                .iter()
                .enumerate()
                .all(|(i, &dl_idx)| i == dl_idx);

        if is_full_refresh_heuristic {
            trace!(
                "Renderer::draw: Full refresh heuristic met. Clearing all with RENDERER_DEFAULT_BG ({:?}).",
                RENDERER_DEFAULT_BG
            );
            driver.clear_all(RENDERER_DEFAULT_BG)?;
            something_was_drawn = true;
        }

        let mut lines_to_draw_content: HashSet<usize> = initially_dirty_lines_from_term;
        let (cursor_abs_x, cursor_abs_y) = term.get_screen_cursor_pos();

        if term.is_cursor_visible() && cursor_abs_y < term_height {
            lines_to_draw_content.insert(cursor_abs_y); // Ensure cursor's line content is redrawn
            something_was_drawn = true;
        }

        let mut sorted_lines_to_draw: Vec<usize> = lines_to_draw_content.into_iter().collect();
        sorted_lines_to_draw.sort_unstable();
        trace!(
            "Renderer::draw: Final lines to process for content: {:?}",
            sorted_lines_to_draw
        );

        for &y_abs in &sorted_lines_to_draw {
            self.draw_line_content(y_abs, term_width, term, driver)?;
        }

        if term.is_cursor_visible() {
            trace!("Renderer::draw: Cursor is visible, calling draw_cursor overlay.");
            self.draw_cursor_overlay(
                cursor_abs_x,
                cursor_abs_y,
                term,
                driver,
                term_width,
                term_height,
            )?;
        } else {
            trace!("Renderer::draw: Cursor is not visible.");
        }

        if something_was_drawn {
            trace!("Renderer::draw: Presenting changes.");
            driver.present()?;
        } else {
            trace!("Renderer::draw: No changes to present.");
        }
        Ok(())
    }

    /// Draws the content of a single line `y_abs` of the terminal.
    /// It iterates through cells, identifying runs of characters with the same style,
    /// runs of spaces, or wide character placeholders, and issues appropriate drawing
    /// commands to the `Driver`.
    fn draw_line_content(
        &self,
        y_abs: usize,
        term_width: usize,
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
    ) -> Result<()> {
        trace!("Renderer::draw_line_content: Drawing line y={}", y_abs);
        let mut current_col: usize = 0;

        while current_col < term_width {
            let start_glyph = term.get_glyph(current_col, y_abs);
            let (eff_fg, eff_bg, eff_flags) = self.get_effective_colors_and_flags(
                start_glyph.attr.fg,
                start_glyph.attr.bg,
                start_glyph.attr.flags,
            );

            let char_for_log = if start_glyph.c == '\0' {
                ' '
            } else {
                start_glyph.c
            }; // Avoid printing null in log
            trace!(
                "  Line {}, Col {}: Start glyph='{}' (attr:{:?}), EffectiveStyle(fg:{:?}, bg:{:?}, flags:{:?})",
                y_abs, current_col, char_for_log, start_glyph.attr, eff_fg, eff_bg, eff_flags
            );

            if start_glyph.c == '\0' {
                // Wide Char Placeholder
                current_col += self.draw_placeholder_cell(current_col, y_abs, eff_bg, driver)?;
            } else if start_glyph.c == ' ' {
                // Potential run of spaces
                let space_run_len = self.draw_space_run(
                    current_col,
                    y_abs,
                    term_width,
                    &start_glyph,
                    term,
                    driver,
                )?;
                if space_run_len > 0 {
                    current_col += space_run_len;
                } else {
                    // Single space that didn't form a run, or some other issue
                    current_col += self.draw_text_segment(
                        current_col,
                        y_abs,
                        term_width,
                        &start_glyph,
                        term,
                        driver,
                    )?;
                }
            } else {
                // Regular text run
                current_col += self.draw_text_segment(
                    current_col,
                    y_abs,
                    term_width,
                    &start_glyph,
                    term,
                    driver,
                )?;
            }
        }
        Ok(())
    }

    /// Handles drawing a placeholder cell (typically for the second half of a wide character).
    /// Returns the number of cells consumed (always 1 for a placeholder).
    fn draw_placeholder_cell(
        &self,
        x: usize,
        y: usize,
        effective_bg: Color,
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let rect = CellRect {
            x,
            y,
            width: 1,
            height: 1,
        };
        trace!(
            "    Line {}, Col {}: Placeholder. FillRect with bg={:?}",
            y, x, effective_bg
        );
        driver.fill_rect(rect, effective_bg)?;
        Ok(1) // Consumed 1 cell
    }

    /// Attempts to identify and draw a run of space characters with consistent background.
    /// Returns the number of cells consumed by the drawn space run.
    fn draw_space_run(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph, // The glyph at start_col, known to be a space
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let (_, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut space_run_len = 0;
        for x_offset in 0..(term_width - start_col) {
            let current_scan_col = start_col + x_offset;
            let glyph_at_scan = term.get_glyph(current_scan_col, y);
            let (_, current_scan_eff_bg, current_scan_flags) = self.get_effective_colors_and_flags(
                glyph_at_scan.attr.fg,
                glyph_at_scan.attr.bg,
                glyph_at_scan.attr.flags,
            );

            if glyph_at_scan.c == ' '
                && current_scan_eff_bg == start_eff_bg
                && current_scan_flags == start_eff_flags
            {
                space_run_len += 1;
            } else {
                break;
            }
        }

        if space_run_len > 0 {
            let rect = CellRect {
                x: start_col,
                y,
                width: space_run_len,
                height: 1,
            };
            trace!(
                "    Line {}, Col {}: Space run found (len {}). FillRect with bg={:?}",
                y, start_col, space_run_len, start_eff_bg
            );
            driver.fill_rect(rect, start_eff_bg)?;
            Ok(space_run_len)
        } else {
            Ok(0) // No run drawn this way
        }
    }

    /// Identifies and draws a run of text characters with consistent styling.
    /// Also handles drawing placeholders for any wide characters within the identified run.
    /// Returns the number of cells consumed by the drawn text run.
    fn draw_text_segment(
        &self,
        start_col: usize,
        y: usize,
        term_width: usize,
        start_glyph: &Glyph, // The glyph at start_col
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
    ) -> Result<usize> {
        let (start_eff_fg, start_eff_bg, start_eff_flags) = self.get_effective_colors_and_flags(
            start_glyph.attr.fg,
            start_glyph.attr.bg,
            start_glyph.attr.flags,
        );

        let mut run_text = String::new();
        let mut run_total_cell_width = 0;
        let mut current_scan_col = start_col;

        while current_scan_col < term_width {
            // Check if adding the next char would exceed available width for the run
            if current_scan_col != start_col && (start_col + run_total_cell_width >= term_width) {
                break; // Run would exceed line boundary
            }

            let glyph_at_scan = term.get_glyph(current_scan_col, y);

            // End run if it's a space or placeholder (these are handled by other dedicated logic paths)
            if glyph_at_scan.c == ' ' || glyph_at_scan.c == '\0' {
                break;
            }

            let (current_glyph_eff_fg, current_glyph_eff_bg, current_glyph_flags) = self
                .get_effective_colors_and_flags(
                    glyph_at_scan.attr.fg,
                    glyph_at_scan.attr.bg,
                    glyph_at_scan.attr.flags,
                );

            if current_glyph_eff_fg == start_eff_fg
                && current_glyph_eff_bg == start_eff_bg
                && current_glyph_flags == start_eff_flags
            {
                let char_display_width = get_char_display_width(glyph_at_scan.c);
                if char_display_width == 0 {
                    trace!(
                        "    Line {}, Col {}: Skipping zero-width char '{}' in text run.",
                        y, current_scan_col, glyph_at_scan.c
                    );
                    // For ZWJ, we must advance the column to avoid infinite loops if many ZWJ follow.
                    // If it's truly zero-width and part of a grapheme cluster handled by the font,
                    // this simple advance might be okay.
                    current_scan_col += 1;
                    continue;
                }
                // Check if this character itself fits
                if start_col + run_total_cell_width + char_display_width > term_width {
                    break;
                }
                run_text.push(glyph_at_scan.c);
                run_total_cell_width += char_display_width;
                current_scan_col += char_display_width; // Advance scan by the cells consumed by this char
            } else {
                break; // Style changed
            }
        }

        if !run_text.is_empty() {
            let coords = CellCoords { x: start_col, y };
            let style = TextRunStyle {
                fg: start_eff_fg,
                bg: start_eff_bg,
                flags: start_eff_flags,
            };
            trace!(
                "    Line {}, Col {}: Text run: '{}' ({} cells). DrawTextRun with style={:?}",
                y, start_col, run_text, run_total_cell_width, style
            );
            driver.draw_text_run(coords, &run_text, style)?;

            // After drawing the text run, iterate through its characters to fill placeholders
            // for any wide characters within that *specific run*.
            let mut col_offset_in_run = 0;
            for ch_in_run in run_text.chars() {
                let char_cell_width = get_char_display_width(ch_in_run);
                if char_cell_width == 2 {
                    let placeholder_x = start_col + col_offset_in_run + 1;
                    if placeholder_x < term_width {
                        let placeholder_rect = CellRect {
                            x: placeholder_x,
                            y,
                            width: 1,
                            height: 1,
                        };
                        trace!(
                            "      Line {}, Col {}: Explicitly filling placeholder for wide char '{}' in run, with bg {:?}",
                            y, placeholder_x, ch_in_run, start_eff_bg
                        );
                        driver.fill_rect(placeholder_rect, start_eff_bg)?;
                    }
                }
                col_offset_in_run += char_cell_width;
            }
            Ok(run_total_cell_width)
        } else {
            // This means the start_glyph itself couldn't form a run (e.g., it was a zero-width char initially)
            // or was a space/placeholder handled by other paths.
            // If it's a non-drawable char, we still need to advance.
            let char_width = get_char_display_width(start_glyph.c);
            if char_width == 0 && start_glyph.c != '\0' {
                // Avoid double-advancing for already handled placeholders
                trace!(
                    "    Line {}, Col {}: Single zero-width char '{}' skipped.",
                    y, start_col, start_glyph.c
                );
                Ok(1) // Advance by 1 logical cell for unhandled zero-width
            } else if char_width > 0 {
                // Should have been caught by run_text logic
                warn!(
                    "    Line {}, Col {}: Single drawable char '{}' did not form text run. This is unexpected.",
                    y, start_col, start_glyph.c
                );
                // Draw it as a run of one to be safe
                let coords = CellCoords { x: start_col, y };
                let style = TextRunStyle {
                    fg: start_eff_fg,
                    bg: start_eff_bg,
                    flags: start_eff_flags,
                };
                driver.draw_text_run(coords, &start_glyph.c.to_string(), style)?;
                if char_width == 2 {
                    if start_col + 1 < term_width {
                        let placeholder_rect = CellRect {
                            x: start_col + 1,
                            y,
                            width: 1,
                            height: 1,
                        };
                        driver.fill_rect(placeholder_rect, start_eff_bg)?;
                    }
                }
                Ok(char_width)
            } else {
                Ok(1) // Default advance if nothing was drawn and it's not a known handled type
            }
        }
    }

    /// Draws the terminal cursor as an overlay.
    ///
    /// The cursor typically inverts the foreground and background colors of the
    /// character cell it is currently on. For wide characters, it correctly
    /// identifies the character to overlay.
    ///
    /// # Arguments
    /// * `cursor_abs_x`, `cursor_abs_y`: The 0-based physical screen coordinates of the cursor.
    /// * `term`: The `TerminalInterface` to get glyph data.
    /// * `driver`: The `Driver` to issue drawing commands.
    /// * `term_width`, `term_height`: Dimensions of the terminal for bounds checking.
    fn draw_cursor_overlay(
        &self,
        cursor_abs_x: usize,
        cursor_abs_y: usize,
        term: &impl TerminalInterface,
        driver: &mut dyn Driver,
        term_width: usize,
        term_height: usize,
    ) -> Result<()> {
        trace!(
            "Renderer::draw_cursor_overlay: Screen cursor pos ({}, {})",
            cursor_abs_x, cursor_abs_y
        );

        // Bounds check should have been done by the caller (Renderer::draw)
        if !(cursor_abs_x < term_width && cursor_abs_y < term_height) {
            warn!(
                "Renderer::draw_cursor_overlay: Cursor at ({}, {}) is out of bounds ({}x{}). Not drawing.",
                cursor_abs_x, cursor_abs_y, term_width, term_height
            );
            return Ok(());
        }

        let physical_cursor_x_for_draw: usize;
        let char_to_draw_at_cursor: char;
        let original_attrs_at_cursor: Attributes;

        let glyph_at_logical_cursor = term.get_glyph(cursor_abs_x, cursor_abs_y);
        let char_for_log1 = if glyph_at_logical_cursor.c == '\0' {
            ' '
        } else {
            glyph_at_logical_cursor.c
        };
        trace!(
            "  Cursor overlay: Glyph at logical cursor pos ({},{}): char='{}', attr={:?}",
            cursor_abs_x, cursor_abs_y, char_for_log1, glyph_at_logical_cursor.attr
        );

        if glyph_at_logical_cursor.c == '\0' && cursor_abs_x > 0 {
            // Cursor is on a wide char placeholder. We draw the cursor over the *start* of the wide char.
            let first_half_glyph = term.get_glyph(cursor_abs_x - 1, cursor_abs_y);
            let char_for_log2 = if first_half_glyph.c == '\0' {
                ' '
            } else {
                first_half_glyph.c
            };
            trace!(
                "    Cursor on placeholder, using first half: char='{}' from col {}",
                char_for_log2,
                cursor_abs_x - 1
            );

            char_to_draw_at_cursor = first_half_glyph.c; // This should be the actual wide char.
            original_attrs_at_cursor = first_half_glyph.attr;
            physical_cursor_x_for_draw = cursor_abs_x - 1;
        } else {
            char_to_draw_at_cursor = glyph_at_logical_cursor.c;
            original_attrs_at_cursor = glyph_at_logical_cursor.attr;
            physical_cursor_x_for_draw = cursor_abs_x;
        }

        let (resolved_original_fg, resolved_original_bg, resolved_original_flags) = self
            .get_effective_colors_and_flags(
                original_attrs_at_cursor.fg,
                original_attrs_at_cursor.bg,
                original_attrs_at_cursor.flags,
            );
        trace!(
            "    Original cell effective attrs for cursor: fg={:?}, bg={:?}, flags={:?}",
            resolved_original_fg, resolved_original_bg, resolved_original_flags
        );

        // Cursor inverts the effective colors of the cell it's on.
        let cursor_char_fg = resolved_original_bg;
        let cursor_cell_bg = resolved_original_fg;
        let cursor_display_flags = resolved_original_flags; // REVERSE flag already handled

        let coords = CellCoords {
            x: physical_cursor_x_for_draw,
            y: cursor_abs_y,
        };
        let style = TextRunStyle {
            fg: cursor_char_fg,
            bg: cursor_cell_bg,
            flags: cursor_display_flags,
        };

        // If the character under the cursor resolved to a NUL (e.g. placeholder at col 0 or bad data), draw a space.
        let final_char_to_draw_for_cursor = if char_to_draw_at_cursor == '\0' {
            ' '
        } else {
            char_to_draw_at_cursor
        };
        trace!(
            "    Drawing cursor overlay: char='{}' at physical ({},{}) with style: {:?}",
            final_char_to_draw_for_cursor, physical_cursor_x_for_draw, cursor_abs_y, style
        );

        // The driver's draw_text_run should handle the full cell width for the character,
        // including for wide characters.
        driver.draw_text_run(coords, &final_char_to_draw_for_cursor.to_string(), style)?;
        Ok(())
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

// Test module remains in src/renderer/tests.rs
#[cfg(test)]
mod tests;
